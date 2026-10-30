def test_overhead_patch_idempotent():
    from solution.overhead import patch

    # patch should run cleanly and be idempotent
    patch()
    patch()
