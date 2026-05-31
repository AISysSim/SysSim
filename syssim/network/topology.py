"""Network topology model: GPUs, switches, links, and route resolution.

Two overlapping connection layers:
  - Intra-node NVLink mesh: per-pair links between GPUs in the same node.
  - Inter-node topology: each GPU has its own uplink to a leaf switch; above
    the leaves, either fat-tree or two-layer multipath spines.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Link:
    """A bidirectional link with capacity and latency."""
    capacity_GBps: float
    latency_us: float


@dataclass
class Switch:
    """A leaf or spine switch."""
    name: str
    kind: str  # "leaf" or "spine"


@dataclass
class Gpu:
    """A GPU host in the topology.

    `node_id` identifies the physical NVLink group (GPUs sharing a node_id
    are NVLink-meshed). `leaf_idx` identifies the routing position (which
    leaf switch this GPU attaches to). For TwoLayerMultiPath these
    coincide (one node per leaf). For Simple and FatTree they differ —
    multiple nodes can share a leaf.

    `nvlink_neighbors` maps peer_rank → Link for intra-node NVLink edges.
    `uplink` is the GPU's link to its leaf switch (None for single-node
    fixtures where no inter-node topology is needed).
    """
    rank: int
    node_id: int
    uplink: Link | None = None
    nvlink_neighbors: dict[int, Link] = field(default_factory=dict)
    leaf_idx: int = -1


@dataclass
class Topology:
    """Graph of GPUs + leaf switches + spine switches + links.

    Path resolution lives on this object: `resolve_path(src, dst, flow_tag)`
    returns the ordered list of links a flow traverses.
    """
    gpus: list[Gpu]
    leaf_switches: list[Switch]
    spine_switches: list[Switch]
    links: list[Link]
    load_balancer_name: str = "ecmp_hash"
    inter_switch_links: dict[tuple[str, str], Link] = field(default_factory=dict)
    # Optional precomputed (src, dst) -> [Link] route table. When non-empty it is the sole
    # source of routes (used by the per-dimension topology builder, whose routing does not fit
    # the nvlink/leaf/spine cases below). Timing still comes entirely from the flow simulator.
    routes: dict[tuple[int, int], list[Link]] = field(default_factory=dict)

    def gpu(self, rank: int) -> Gpu:
        """Return the Gpu with the given rank."""
        return self.gpus[rank]

    def resolve_path(self, src: int, dst: int, flow_tag: int) -> list[Link]:
        """Return the ordered list of links a flow traverses from `src` to `dst`.

        A precomputed `routes` table (if present) takes precedence. Otherwise three cases:
          - NVLink peers (same node): single NVLink edge.
          - Same leaf, different nodes (Simple, FatTree intra-pod): two
            uplinks meeting at the shared leaf — no spine traversal.
          - Different leaves: uplink_src → leaf↔spine → spine↔leaf_dst →
            uplink_dst, with the load balancer picking the spine.
        """
        if self.routes:
            return self.routes.get((src, dst), [])
        from .load_balancer import get_load_balancer
        g_src = self.gpu(src)
        g_dst = self.gpu(dst)
        if dst in g_src.nvlink_neighbors:
            return [g_src.nvlink_neighbors[dst]]
        leaf_src = self.leaf_switches[g_src.leaf_idx]
        leaf_dst = self.leaf_switches[g_dst.leaf_idx]
        if leaf_src is leaf_dst:
            return [g_src.uplink, g_dst.uplink]
        lb = get_load_balancer(self.load_balancer_name)
        spine_idx = lb.choose_spine(
            src=src, dst=dst, flow_tag=flow_tag,
            num_spines=len(self.spine_switches),
        )
        spine = self.spine_switches[spine_idx]
        return [
            g_src.uplink,
            self.inter_switch_links[(leaf_src.name, spine.name)],
            self.inter_switch_links[(spine.name, leaf_dst.name)],
            g_dst.uplink,
        ]


def build_two_layer_multipath(
    *,
    num_racks: int,
    nodes_per_rack: int,
    gpus_per_node: int,
    num_spines: int,
    per_gpu_bandwidth_GBps: float,
    uplink_bandwidth_GBps: float,
    intra_node_bandwidth_GBps: float,
    intra_node_latency_us: float = 0.0,
    inter_node_latency_us: float = 0.0,
    load_balancer_name: str = "ecmp_hash",
) -> Topology:
    """Build a two-layer multipath topology with ECMP between leaves and spines.

    `num_racks` racks; each rack has `nodes_per_rack` nodes sharing one ToR
    (leaf switch). Each node has `gpus_per_node` GPUs in a complete NVLink
    mesh. Each GPU has its own per-NIC uplink (capacity
    `per_gpu_bandwidth_GBps`) to its rack's ToR. Each leaf is connected to
    every spine by an `uplink_bandwidth_GBps` link. Total GPUs =
    num_racks * nodes_per_rack * gpus_per_node.
    """
    leaf_switches = [Switch(name=f"leaf-{r}", kind="leaf") for r in range(num_racks)]
    spine_switches = [Switch(name=f"spine-{s}", kind="spine") for s in range(num_spines)]
    links: list[Link] = []
    gpus: list[Gpu] = []

    # Per-GPU uplinks (one Link object per GPU)
    for r in range(num_racks):
        for node_in_rack in range(nodes_per_rack):
            node_id = r * nodes_per_rack + node_in_rack
            for local in range(gpus_per_node):
                rank = node_id * gpus_per_node + local
                uplink = Link(capacity_GBps=per_gpu_bandwidth_GBps,
                              latency_us=inter_node_latency_us)
                links.append(uplink)
                gpus.append(Gpu(rank=rank, node_id=node_id, leaf_idx=r,
                                uplink=uplink))

    # NVLink mesh within each node
    for node_id in range(num_racks * nodes_per_rack):
        node_ranks = [node_id * gpus_per_node + i for i in range(gpus_per_node)]
        for i in node_ranks:
            for j in node_ranks:
                if i == j or j in gpus[i].nvlink_neighbors:
                    continue
                nvlink = Link(capacity_GBps=intra_node_bandwidth_GBps,
                              latency_us=intra_node_latency_us)
                links.append(nvlink)
                gpus[i].nvlink_neighbors[j] = nvlink
                gpus[j].nvlink_neighbors[i] = nvlink

    # leaf↔spine links: one per (leaf, spine) pair
    inter_switch_links: dict[tuple[str, str], Link] = {}
    for leaf in leaf_switches:
        for spine in spine_switches:
            uplink_to_spine = Link(capacity_GBps=uplink_bandwidth_GBps,
                                   latency_us=inter_node_latency_us)
            links.append(uplink_to_spine)
            inter_switch_links[(leaf.name, spine.name)] = uplink_to_spine
            inter_switch_links[(spine.name, leaf.name)] = uplink_to_spine

    return Topology(
        gpus=gpus,
        leaf_switches=leaf_switches,
        spine_switches=spine_switches,
        links=links,
        load_balancer_name=load_balancer_name,
        inter_switch_links=inter_switch_links,
    )


def build_fat_tree(
    *,
    k: int,
    gpus_per_node: int,
    per_gpu_bandwidth_GBps: float,
    intra_node_bandwidth_GBps: float,
    oversub_ratio: float = 1.0,
    intra_node_latency_us: float = 0.0,
    inter_node_latency_us: float = 0.0,
    load_balancer_name: str = "ecmp_hash",
) -> Topology:
    """Build a k-ary fat-tree topology.

    A k-ary fat-tree has k pods, each with k/2 edge switches and k/2 agg
    switches. There are (k/2)^2 core (spine) switches. Each edge switch
    connects to k/2 hosts. Total hosts = k^3/4.

    `gpus_per_node` GPUs share one "host" (one leaf switch position),
    so total GPUs = (k^3 / 4) * gpus_per_node.
    """
    if k % 2 != 0:
        raise ValueError(f"k must be even for FatTree, got {k}")
    pods = k
    edges_per_pod = k // 2
    hosts_per_edge = k // 2
    num_hosts = pods * edges_per_pod * hosts_per_edge
    num_cores = (k // 2) ** 2

    # Each host's edge switch becomes a "leaf switch" in our model.
    leaf_switches = [Switch(name=f"edge-p{p}-e{e}", kind="leaf")
                     for p in range(pods)
                     for e in range(edges_per_pod)]
    # Core switches are our "spine_switches".
    spine_switches = [Switch(name=f"core-{c}", kind="spine")
                      for c in range(num_cores)]

    gpus: list[Gpu] = []
    links: list[Link] = []
    for host_idx in range(num_hosts):
        pod_idx = host_idx // (edges_per_pod * hosts_per_edge)
        edge_idx = (host_idx // hosts_per_edge) % edges_per_pod
        leaf_idx = pod_idx * edges_per_pod + edge_idx
        # gpus_per_node GPUs share this host's leaf switch
        for local in range(gpus_per_node):
            rank = host_idx * gpus_per_node + local
            uplink = Link(capacity_GBps=per_gpu_bandwidth_GBps / oversub_ratio,
                          latency_us=inter_node_latency_us)
            links.append(uplink)
            gpus.append(Gpu(rank=rank, node_id=host_idx, leaf_idx=leaf_idx, uplink=uplink))

    # NVLink mesh per host
    for host_idx in range(num_hosts):
        node_ranks = [host_idx * gpus_per_node + i for i in range(gpus_per_node)]
        for i in node_ranks:
            for j in node_ranks:
                if i == j or j in gpus[i].nvlink_neighbors:
                    continue
                nvlink = Link(capacity_GBps=intra_node_bandwidth_GBps,
                              latency_us=intra_node_latency_us)
                links.append(nvlink)
                gpus[i].nvlink_neighbors[j] = nvlink
                gpus[j].nvlink_neighbors[i] = nvlink

    # edge↔core (we treat the agg layer as transparent for path resolution v1;
    # the load balancer picks one of num_cores spines and the path is
    # edge_src → core → edge_dst)
    inter_switch_links: dict[tuple[str, str], Link] = {}
    for leaf in leaf_switches:
        for spine in spine_switches:
            link = Link(capacity_GBps=per_gpu_bandwidth_GBps / oversub_ratio,
                        latency_us=inter_node_latency_us)
            links.append(link)
            inter_switch_links[(leaf.name, spine.name)] = link
            inter_switch_links[(spine.name, leaf.name)] = link

    return Topology(
        gpus=gpus,
        leaf_switches=leaf_switches,
        spine_switches=spine_switches,
        links=links,
        load_balancer_name=load_balancer_name,
        inter_switch_links=inter_switch_links,
    )


def build_arbitrary(
    *,
    num_racks: int,
    nodes_per_rack: int,
    gpus_per_node: int,
    per_gpu_bandwidth_GBps: float,
    rack_bandwidth_GBps: float,
    intra_node_bandwidth_GBps: float,
    intra_node_latency_us: float = 0.0,
    inter_node_latency_us: float = 0.0,
    load_balancer_name: str = "ecmp_hash",
) -> Topology:
    """Arbitrary 2-tier topology: per-GPU NIC uplinks into a rack/root switch tree.

    `num_racks` racks under a single root switch. Each rack has `nodes_per_rack`
    nodes; each node has `gpus_per_node` GPUs in a complete NVLink mesh.
    Each GPU has its own per-NIC uplink to its rack's ToR; each ToR has a
    single uplink (capacity `rack_bandwidth_GBps`) to the root.

    Path resolution:
      - Same node → NVLink.
      - Same rack, different nodes → host_uplink → ToR → host_uplink.
      - Different racks → host_uplink → ToR → root → ToR → host_uplink.

    For users who only know intra-node and per-node inter-node bandwidth
    (typical cloud rental), use `build_simple` instead.
    """
    root = Switch(name="root", kind="spine")
    spine_switches = [root]
    leaf_switches = [Switch(name=f"tor-{r}", kind="leaf") for r in range(num_racks)]
    links: list[Link] = []
    gpus: list[Gpu] = []

    # Per-GPU uplinks
    for r in range(num_racks):
        for node_in_rack in range(nodes_per_rack):
            node_id = r * nodes_per_rack + node_in_rack
            for local in range(gpus_per_node):
                rank = node_id * gpus_per_node + local
                uplink = Link(capacity_GBps=per_gpu_bandwidth_GBps,
                              latency_us=inter_node_latency_us)
                links.append(uplink)
                gpus.append(Gpu(rank=rank, node_id=node_id, leaf_idx=r,
                                uplink=uplink))

    # NVLink mesh per node
    for r in range(num_racks):
        for node_in_rack in range(nodes_per_rack):
            node_id = r * nodes_per_rack + node_in_rack
            node_ranks = [node_id * gpus_per_node + i for i in range(gpus_per_node)]
            for i in node_ranks:
                for j in node_ranks:
                    if i == j or j in gpus[i].nvlink_neighbors:
                        continue
                    nvlink = Link(capacity_GBps=intra_node_bandwidth_GBps,
                                  latency_us=intra_node_latency_us)
                    links.append(nvlink)
                    gpus[i].nvlink_neighbors[j] = nvlink
                    gpus[j].nvlink_neighbors[i] = nvlink

    # leaf ↔ root: one link per ToR (single-path)
    inter_switch_links: dict[tuple[str, str], Link] = {}
    for leaf in leaf_switches:
        link = Link(capacity_GBps=rack_bandwidth_GBps,
                    latency_us=inter_node_latency_us)
        links.append(link)
        inter_switch_links[(leaf.name, root.name)] = link
        inter_switch_links[(root.name, leaf.name)] = link

    return Topology(
        gpus=gpus,
        leaf_switches=leaf_switches,
        spine_switches=spine_switches,
        links=links,
        load_balancer_name=load_balancer_name,
        inter_switch_links=inter_switch_links,
    )


def build_simple(
    *,
    num_nodes: int,
    gpus_per_node: int,
    intra_node_bandwidth_GBps: float,
    inter_node_bandwidth_GBps: float,
    intra_node_latency_us: float = 0.0,
    inter_node_latency_us: float = 0.0,
) -> Topology:
    """Simplest topology — one NIC per node, flat interconnect above.

    Each of `num_nodes` nodes has `gpus_per_node` GPUs in NVLink mesh. All
    GPUs in a node SHARE one node-level NIC of `inter_node_bandwidth_GBps`
    (the typical cloud-rental model: one VM = one NIC, shared across the
    GPUs you rent). The interconnect above the NICs is modeled as
    non-bottlenecking — only the per-node NIC matters for inter-node
    bandwidth.

    Use this when you only know intra-node and inter-node bandwidth and
    don't want to think about rack structure. For explicit
    rack / per-GPU-NIC modeling, use `build_arbitrary` or
    `build_two_layer_multipath` instead.
    """
    root = Switch(name="root", kind="spine")
    spine_switches = [root]
    leaf_switches = [Switch(name=f"tor-{n}", kind="leaf") for n in range(num_nodes)]
    links: list[Link] = []
    gpus: list[Gpu] = []

    # One shared NIC per node; all GPUs in the node use it as their uplink.
    for n in range(num_nodes):
        nic = Link(capacity_GBps=inter_node_bandwidth_GBps,
                   latency_us=inter_node_latency_us)
        links.append(nic)
        for g in range(gpus_per_node):
            rank = n * gpus_per_node + g
            gpus.append(Gpu(rank=rank, node_id=n, leaf_idx=n, uplink=nic))

    # NVLink mesh per node
    for n in range(num_nodes):
        node_ranks = [n * gpus_per_node + i for i in range(gpus_per_node)]
        for i in node_ranks:
            for j in node_ranks:
                if i == j or j in gpus[i].nvlink_neighbors:
                    continue
                nvlink = Link(capacity_GBps=intra_node_bandwidth_GBps,
                              latency_us=intra_node_latency_us)
                links.append(nvlink)
                gpus[i].nvlink_neighbors[j] = nvlink
                gpus[j].nvlink_neighbors[i] = nvlink

    # leaf ↔ root: one link per ToR, sized to the per-node NIC so it never
    # bottlenecks above the NIC layer (incoming many-to-one converges
    # equally on the dst NIC and its leaf↔root, giving the same per-flow
    # share either way).
    inter_switch_links: dict[tuple[str, str], Link] = {}
    for leaf in leaf_switches:
        link = Link(capacity_GBps=inter_node_bandwidth_GBps,
                    latency_us=inter_node_latency_us)
        links.append(link)
        inter_switch_links[(leaf.name, root.name)] = link
        inter_switch_links[(root.name, leaf.name)] = link

    return Topology(
        gpus=gpus,
        leaf_switches=leaf_switches,
        spine_switches=spine_switches,
        links=links,
        load_balancer_name="ecmp_hash",  # unused (single spine)
        inter_switch_links=inter_switch_links,
    )


def build_dimensional(*, dims, size, bandwidth, latency_ns,
                      gpus_per_node=None, load_balancer_name: str = "ecmp_hash") -> Topology:
    """Build a Topology from an ordered, per-dimension network description.

    Each dimension `d` connects the GPUs that share all *other* coordinates using the topology
    `dims[d]` (one of "switch", "ring", "fully_connected") with per-dimension `size[d]` endpoints,
    link `bandwidth[d]` (GB/s) and `latency_ns[d]` (ns). A rank decomposes into per-dimension
    coordinates (dimension 0 varies fastest); a flow is routed one differing dimension at a time,
    concatenating the per-dimension link segments. All timing is left to the flow simulator over
    the resulting links — this builder only lays down links and the route table.

      switch          : a switch per group; every member gets an uplink (the shared-egress case).
      fully_connected : a dedicated link per member pair.
      ring            : a link between adjacent members; routes take the shorter arc.
    """
    ndim = len(dims)
    if not (ndim == len(size) == len(bandwidth) == len(latency_ns)):
        raise ValueError("per-dimension topology arrays must have equal length")
    radix = [int(n) for n in size]
    total = 1
    for n in radix:
        total *= n

    def coords(r: int) -> tuple:
        c, x = [], r
        for n in radix:
            c.append(x % n)
            x //= n
        return tuple(c)

    def rank_of(c) -> int:
        r, mul = 0, 1
        for d, n in enumerate(radix):
            r += c[d] * mul
            mul *= n
        return r

    coord = [coords(r) for r in range(total)]
    links: list[Link] = []
    switches: list[Switch] = []
    gpn = int(gpus_per_node) if gpus_per_node else total
    gpus = [Gpu(rank=r, node_id=r // gpn) for r in range(total)]

    fc_link: dict = {}       # (d, a, b) -> Link
    sw_send: dict = {}       # (d, a) -> Link (a -> switch; full-duplex egress)
    sw_recv: dict = {}       # (d, a) -> Link (switch -> a; full-duplex ingress)
    ring_links: dict = {}    # (d, group_key) -> [(a, b, Link)] in ring order

    for d in range(ndim):
        typ, agg_bw = dims[d], float(bandwidth[d])
        lat_us = float(latency_ns[d]) / 1000.0
        # `bandwidth[d]` is the per-NODE UNI-DIRECTIONAL aggregate in this dimension (datasheet, e.g.
        # GH200 NVLink is 900 GB/s/GPU bidirectional -> 450 GB/s/GPU per direction). The per-LINK
        # capacity is that aggregate divided by the node's degree: fully_connected -> (size-1) peer
        # links, ring -> 2 neighbours, switch -> 1 uplink. Links are FULL-DUPLEX -- each direction is
        # an INDEPENDENT directed link of this capacity -- so a 4-way NVLink mesh's per-peer
        # per-direction link is 450/3 = 150 GB/s, derived here rather than hand-entered.
        n_d = radix[d]
        if typ == "fully_connected":
            degree = max(1, n_d - 1)
        elif typ == "ring":
            degree = 2 if n_d >= 3 else 1
        else:  # switch: one uplink per node
            degree = 1
        bw = agg_bw / degree
        groups: dict = {}
        for r in range(total):
            groups.setdefault(coord[r][:d] + coord[r][d + 1:], []).append(r)
        for key, members in groups.items():
            members.sort(key=lambda r: coord[r][d])
            if typ == "fully_connected":
                # Full-duplex: a dedicated DIRECTED link per ordered pair (a->b is independent of
                # b->a), so opposing flows do not contend.
                for a in members:
                    for b in members:
                        if a == b:
                            continue
                        lk = Link(capacity_GBps=bw, latency_us=lat_us)
                        links.append(lk)
                        fc_link[(d, a, b)] = lk
            elif typ == "switch":
                switches.append(Switch(name=f"sw_d{d}_{key}", kind="leaf"))
                for a in members:
                    # Full-duplex: independent send (a->switch) and recv (switch->a) directed links,
                    # so a->b and b->a do not contend (real NICs send/recv independently, matching
                    # the fully_connected convention above). Sharing one link would halve the rate.
                    snd = Link(capacity_GBps=bw, latency_us=lat_us)
                    rcv = Link(capacity_GBps=bw, latency_us=lat_us)
                    links.append(snd)
                    links.append(rcv)
                    sw_send[(d, a)] = snd
                    sw_recv[(d, a)] = rcv
            elif typ == "ring":
                rl, m = [], len(members)
                for i in range(m):
                    a, b = members[i], members[(i + 1) % m]
                    lk = Link(capacity_GBps=bw, latency_us=lat_us)
                    links.append(lk)
                    rl.append((a, b, lk))
                ring_links[(d, key)] = rl
            else:
                raise ValueError(f"unknown per-dimension topology: {typ!r}")

    def seg(d: int, a: int, b: int) -> list[Link]:
        """Links for a single-dimension hop a->b (a, b share every coordinate but dim d)."""
        typ = dims[d]
        if typ == "fully_connected":
            return [fc_link[(d, a, b)]]
        if typ == "switch":
            return [sw_send[(d, a)], sw_recv[(d, b)]]
        rl = ring_links[(d, coord[a][:d] + coord[a][d + 1:])]
        order = [x[0] for x in rl]
        ia, ib, m = order.index(a), order.index(b), len(order)
        out, i = [], ia
        if (ib - ia) % m <= (ia - ib) % m:          # forward arc is shorter
            while i != ib:
                out.append(rl[i][2]); i = (i + 1) % m
        else:                                        # backward arc
            while i != ib:
                j = (i - 1) % m
                out.append(rl[j][2]); i = j
        return out

    routes: dict = {}
    for src in range(total):
        for dst in range(total):
            if src == dst:
                continue
            cur, c, out = src, list(coord[src]), []
            for d in range(ndim):
                if c[d] == coord[dst][d]:
                    continue
                c[d] = coord[dst][d]
                nxt = rank_of(tuple(c))
                out += seg(d, cur, nxt)
                cur = nxt
            routes[(src, dst)] = out

    return Topology(gpus=gpus, leaf_switches=switches, spine_switches=[], links=links,
                    load_balancer_name=load_balancer_name, routes=routes)


def build_topology_from_config(hw) -> Topology:
    """Build a Topology from a HardwareConfig's per-dimension `topology` block.

    The block is an ordered per-dimension description — `dims` (topology type per dimension:
    switch | ring | fully_connected), `size`, `bandwidth` (GB/s) and `latency` (ns) — expanded
    into explicit links by `build_dimensional` and timed by the flow simulator. `gpus_per_node`
    stays on `HardwareConfig` since it is also a compute/placement concern. (The `build_simple` /
    `build_two_layer_multipath` / `build_fat_tree` / `build_arbitrary` constructors remain as
    Python builders for direct use; they are no longer reachable from a hardware YAML.)
    """
    if hw.topology is None:
        raise ValueError(
            "HardwareConfig.topology is required for the network simulator"
        )
    f = dict(hw.topology)
    if "dims" not in f:
        raise ValueError(
            "topology block must use the per-dimension format "
            "(dims / size / bandwidth / latency)"
        )
    n = len(f["dims"])
    return build_dimensional(
        dims=list(f["dims"]),
        size=list(f["size"]),
        bandwidth=list(f["bandwidth"]),
        latency_ns=list(f.get("latency", [0.0] * n)),
        gpus_per_node=hw.gpus_per_node,
        load_balancer_name=f.get("load_balancer", "ecmp_hash"),
    )
