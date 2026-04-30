def test_import():
    import flashquest  # noqa: F401


def test_submodules_importable():
    from flashquest import kernel, cache, model, runtime  # noqa: F401
