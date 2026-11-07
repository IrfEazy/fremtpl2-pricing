import pytest

from fremtpl2_pricing.arff import read_arff


def test_read_arff(tmp_path):
    f = tmp_path / "t.arff"
    f.write_text(
        "@relation t\n@attribute id numeric\n@attribute Area {'A','B'}\n"
        "@attribute Gas string\n@data\n1,'A','Regular'\n2,'B','Diesel'\n"
    )
    df = read_arff(f)
    assert list(df.columns) == ["id", "Area", "Gas"]
    assert df["id"].tolist() == [1, 2]
    assert list(df["Area"].cat.categories) == ["A", "B"]
    assert df["Gas"].tolist() == ["Regular", "Diesel"]


def _write(tmp_path, ids):
    f = tmp_path / "i.arff"
    body = "".join(f"{i},1\n" for i in ids)
    f.write_text("@relation t\n@attribute IDpol numeric\n@attribute x numeric\n@data\n" + body)
    return f


def test_int_columns_become_int64(tmp_path):
    df = read_arff(_write(tmp_path, ["1.0", "2", "3000000000"]), int_columns=["IDpol"])
    assert str(df["IDpol"].dtype) == "int64"
    assert df["IDpol"].tolist() == [1, 2, 3000000000]


def test_int_columns_fail_on_non_integral(tmp_path):
    with pytest.raises(ValueError, match="non-integral"):
        read_arff(_write(tmp_path, ["1", "2.5"]), int_columns=["IDpol"])


def test_int_columns_fail_on_missing_value_and_unknown_column(tmp_path):
    with pytest.raises(ValueError, match="missing"):
        read_arff(_write(tmp_path, ["1", "?"]), int_columns=["IDpol"])
    with pytest.raises(ValueError, match="not in file"):
        read_arff(_write(tmp_path, ["1"]), int_columns=["nope"])


def test_numeric_without_int_columns_stays_float(tmp_path):
    assert str(read_arff(_write(tmp_path, ["1.0"]))["IDpol"].dtype) == "float64"
