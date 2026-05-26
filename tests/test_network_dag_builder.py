"""Unit tests for DAG construction with dependency inference.

Tests cover:
- Rule 1: Data dependency (receive-before-send)
- Rule 2: Send serialization (single-threaded NIC)
- No duplicate dependencies
- Tag generation
"""

from syssim.network import build_dag


class TestBuildDAG:
    """Test automatic dependency inference in build_dag."""

    def test_empty_steps(self):
        """Empty step list produces empty op list."""
        ops = build_dag([])
        assert ops == []

    def test_single_step_no_dependencies(self):
        """Ops in first step have no dependencies."""
        steps = [
            [(0, 1, 100), (1, 2, 100), (2, 0, 100)],
        ]
        ops = build_dag(steps)

        assert len(ops) == 3
        for op in ops:
            assert op.deps == []

    def test_rule_1_data_dependency(self):
        """Rule 1: Send FROM rank r depends on receives TO rank r in previous step.

        Example: 2-rank exchange
        - Step 0: 0->1 (100 bytes), 1->0 (100 bytes)
        - Step 1: 0->1 (100 bytes), 1->0 (100 bytes)

        At step 1:
        - 0->1 depends on 1->0 from step 0 (rank 0 received from rank 1)
        - 1->0 depends on 0->1 from step 0 (rank 1 received from rank 0)
        """
        steps = [
            [(0, 1, 100), (1, 0, 100)],  # Step 0
            [(0, 1, 100), (1, 0, 100)],  # Step 1
        ]
        ops = build_dag(steps)

        assert len(ops) == 4

        # Step 0 ops (indices 0, 1)
        op_0_to_1_step0 = ops[0]
        op_1_to_0_step0 = ops[1]
        assert op_0_to_1_step0.deps == []
        assert op_1_to_0_step0.deps == []

        # Step 1 ops (indices 2, 3)
        op_0_to_1_step1 = ops[2]
        op_1_to_0_step1 = ops[3]

        # Rule 1: 0->1 at step 1 depends on 1->0 at step 0
        # (rank 0 sends data it received from rank 1)
        assert op_1_to_0_step0 in op_0_to_1_step1.deps

        # Rule 1: 1->0 at step 1 depends on 0->1 at step 0
        # (rank 1 sends data it received from rank 0)
        assert op_0_to_1_step0 in op_1_to_0_step1.deps

    def test_rule_2_send_serialization(self):
        """Rule 2: Send FROM rank r depends on previous send FROM rank r.

        Example: Rank 0 sends to multiple destinations sequentially
        - Step 0: 0->1
        - Step 1: 0->2

        0->2 at step 1 depends on 0->1 at step 0 (NIC serialization)
        """
        steps = [
            [(0, 1, 100)],  # Step 0
            [(0, 2, 100)],  # Step 1
        ]
        ops = build_dag(steps)

        assert len(ops) == 2

        op_0_to_1 = ops[0]
        op_0_to_2 = ops[1]

        assert op_0_to_1.deps == []
        # Rule 2: 0->2 depends on previous send from rank 0
        assert op_0_to_1 in op_0_to_2.deps

    def test_both_rules_combined(self):
        """Both rules apply when rank receives then sends multiple times.

        Example: 3-rank ring reduce-scatter pattern
        - Step 0: 0->1, 1->2, 2->0 (each rank sends chunk_0)
        - Step 1: 0->1, 1->2, 2->0 (each rank sends chunk_1)

        At step 1:
        - 0->1 depends on:
            * 2->0 from step 0 (Rule 1: rank 0 received from rank 2)
            * 0->1 from step 0 (Rule 2: rank 0's previous send)
        """
        steps = [
            [(0, 1, 100), (1, 2, 100), (2, 0, 100)],  # Step 0
            [(0, 1, 100), (1, 2, 100), (2, 0, 100)],  # Step 1
        ]
        ops = build_dag(steps)

        assert len(ops) == 6

        # Step 1 op: 0->1 (index 3)
        op_0_to_1_step1 = ops[3]

        # From step 0
        op_0_to_1_step0 = ops[0]  # 0->1 at step 0
        op_2_to_0_step0 = ops[2]  # 2->0 at step 0

        # Should have 2 dependencies
        assert len(op_0_to_1_step1.deps) == 2

        # Rule 1: Depends on 2->0 (data dependency)
        assert op_2_to_0_step0 in op_0_to_1_step1.deps

        # Rule 2: Depends on 0->1 from previous step (send serialization)
        assert op_0_to_1_step0 in op_0_to_1_step1.deps

    def test_no_duplicate_dependencies(self):
        """Avoid duplicate deps when Rule 1 and Rule 2 point to same op.

        Edge case: If rank r sends to rank s at step 0, and receives from rank s
        at step 0, then sends to rank s again at step 1, both rules point to
        the same predecessor.

        Example:
        - Step 0: 0->1, 1->0
        - Step 1: 0->1

        At step 1, 0->1 depends on:
        - Rule 1: 1->0 from step 0 (rank 0 received from rank 1)
        - Rule 2: 0->1 from step 0 (rank 0's previous send)

        These are different ops, so no duplicate here. But if they were the same,
        we should deduplicate.
        """
        # Construct a case where both rules could apply
        # Actually, with current logic, Rule 1 and Rule 2 always point to different ops
        # because Rule 1 looks at "sends TO src" and Rule 2 looks at "sends FROM src"
        # So this test verifies that deps list doesn't have duplicates in general

        steps = [
            [(0, 1, 100), (1, 0, 100)],
            [(0, 1, 100)],
        ]
        ops = build_dag(steps)

        op_0_to_1_step1 = ops[2]

        # Check no duplicate dependencies (check object identity)
        deps = op_0_to_1_step1.deps
        for i, dep1 in enumerate(deps):
            for j, dep2 in enumerate(deps):
                if i != j:
                    assert dep1 is not dep2, "Dependencies should be unique"

    def test_multiple_senders_to_same_rank(self):
        """Multiple ranks can send to same destination concurrently.

        Example: Gather pattern (all ranks send to rank 0)
        - Step 0: 1->0, 2->0, 3->0

        All three ops have no dependencies (they're concurrent).
        """
        steps = [
            [(1, 0, 100), (2, 0, 100), (3, 0, 100)],
        ]
        ops = build_dag(steps)

        assert len(ops) == 3
        for op in ops:
            assert op.deps == []

    def test_broadcast_pattern(self):
        """Binomial tree broadcast has correct dependencies.

        Example: 4-rank broadcast from rank 0
        - Step 0: 0->1
        - Step 1: 0->2, 1->3

        Dependencies:
        - 0->2 depends on 0->1 (Rule 2: send serialization from rank 0)
        - 1->3 depends on 0->1 (Rule 1: rank 1 received from rank 0)
        """
        steps = [
            [(0, 1, 100)],  # Step 0
            [(0, 2, 100), (1, 3, 100)],  # Step 1
        ]
        ops = build_dag(steps)

        assert len(ops) == 3

        op_0_to_1 = ops[0]
        op_0_to_2 = ops[1]
        op_1_to_3 = ops[2]

        # Step 0
        assert op_0_to_1.deps == []

        # Step 1
        assert op_0_to_1 in op_0_to_2.deps  # Rule 2
        assert op_0_to_1 in op_1_to_3.deps  # Rule 1

    def test_tag_generation_with_prefix(self):
        """Tags are generated with prefix and step index."""
        steps = [
            [(0, 1, 100)],
            [(1, 2, 100)],
        ]
        ops = build_dag(steps, tag_prefix="allreduce")

        assert ops[0].tag == "allreduce_step_0"
        assert ops[1].tag == "allreduce_step_1"

    def test_tag_generation_without_prefix(self):
        """Tags default to step_N when no prefix given."""
        steps = [
            [(0, 1, 100)],
            [(1, 2, 100)],
        ]
        ops = build_dag(steps)

        assert ops[0].tag == "step_0"
        assert ops[1].tag == "step_1"

    def test_op_fields(self):
        """Op objects have correct field values."""
        steps = [
            [(0, 1, 12345)],
        ]
        ops = build_dag(steps, tag_prefix="test")

        op = ops[0]
        assert op.src == 0
        assert op.dst == 1
        assert op.size == 12345
        assert op.tag == "test_step_0"
        assert op.deps == []
        # Simulation fields initialized to 0
        assert op.remaining_bytes == 0.0
        assert op.start_time == 0.0
        assert op.finish_time == 0.0

    def test_complex_pattern_three_steps(self):
        """Complex pattern with 3 steps exercises all dependency logic.

        Pattern: 4-rank ring
        - Step 0: 0->1, 1->2, 2->3, 3->0
        - Step 1: 0->1, 1->2, 2->3, 3->0
        - Step 2: 0->1, 1->2, 2->3, 3->0

        Each op in step N depends on 2 ops from step N-1:
        - The receive TO its src (Rule 1)
        - The previous send FROM its src (Rule 2)
        """
        steps = [
            [(0, 1, 100), (1, 2, 100), (2, 3, 100), (3, 0, 100)],
            [(0, 1, 100), (1, 2, 100), (2, 3, 100), (3, 0, 100)],
            [(0, 1, 100), (1, 2, 100), (2, 3, 100), (3, 0, 100)],
        ]
        ops = build_dag(steps)

        assert len(ops) == 12

        # Check step 0 (indices 0-3): no dependencies
        for i in range(4):
            assert ops[i].deps == []

        # Check step 1 (indices 4-7): each has 2 dependencies
        for i in range(4, 8):
            assert len(ops[i].deps) == 2

        # Check step 2 (indices 8-11): each has 2 dependencies
        for i in range(8, 12):
            assert len(ops[i].deps) == 2

        # Spot check: op at index 8 (0->1 at step 2)
        # Should depend on:
        # - 3->0 from step 1 (index 7) - Rule 1
        # - 0->1 from step 1 (index 4) - Rule 2
        assert ops[7] in ops[8].deps
        assert ops[4] in ops[8].deps
