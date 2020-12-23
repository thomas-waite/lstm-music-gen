from prepare_data import PrepareData


def test_extract_notes():
    data_preparation = PrepareData()
    notes = data_preparation.get_notes()
    assert len(notes) > 1
