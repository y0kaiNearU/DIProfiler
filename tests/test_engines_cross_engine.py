"""
Regression coverage for writers that assumed the incoming nw.LazyFrame's
native object exposes its own .to_arrow() (true for Polars/DuckDB natives,
false for pandas and DataFusion-bridged pyarrow.Table natives). Loading with
one engine and writing with another is an explicitly supported pattern
(FrameLoader/FrameWriter can be given different engines), so every writer
must accept a frame regardless of which engine produced it.
"""
import pytest

from engines.datafusion.loader import DataFusionLoader
from engines.datafusion.writer import DataFusionWriter
from engines.duckdb.loader import DuckDBLoader
from engines.duckdb.writer import DuckDBWriter
from engines.pandas.loader import PandasLoader
from engines.pandas.writer import PandasWriter
from engines.polars.loader import PolarsLoader
from engines.polars.writer import PolarsWriter
from models.models import DatasetInfo, FileFormat, FileSource, PipelineRequest

LOADERS = {
    "duckdb": DuckDBLoader,
    "polars": PolarsLoader,
    "pandas": PandasLoader,
    "datafusion": DataFusionLoader,
}
WRITERS = {
    "duckdb": DuckDBWriter,
    "polars": PolarsWriter,
    "pandas": PandasWriter,
    "datafusion": DataFusionWriter,
}


def _load_req(path, fmt=FileFormat.CSV):
    return PipelineRequest(source=DatasetInfo(source=FileSource(path=path, format=fmt)))


def _write_req(src_path, dst_path):
    return PipelineRequest(
        source=DatasetInfo(source=FileSource(path=src_path, format=FileFormat.CSV)),
        destination=DatasetInfo(source=FileSource(path=dst_path, format=FileFormat.PARQUET)),
    )


@pytest.mark.parametrize("load_engine", sorted(LOADERS))
@pytest.mark.parametrize("write_engine", sorted(WRITERS))
def test_load_with_one_engine_write_with_another(tmp_path, load_engine, write_engine):
    src = tmp_path / "in.csv"
    src.write_text("a,b\n1,x\n2,y\n")
    dst = tmp_path / f"out_{load_engine}_{write_engine}.parquet"

    frame = LOADERS[load_engine]().load(_load_req(str(src)))
    WRITERS[write_engine]().write(frame, _write_req(str(src), str(dst)))

    assert dst.exists() or dst.is_dir()

    result = DuckDBLoader().load(_load_req(str(dst), fmt=FileFormat.PARQUET))
    rows = sorted(result.collect().rows(named=True), key=lambda r: r["a"])
    assert rows == [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]
