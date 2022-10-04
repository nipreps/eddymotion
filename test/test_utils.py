import ants


def test_antsimage_from_path():
    """"""
    # Generate and save test data to file
    # Send through antsimage_from_path
    # Check if ANTsImage, same as original data
    assert isinstance(result, ants.ANTsImage)