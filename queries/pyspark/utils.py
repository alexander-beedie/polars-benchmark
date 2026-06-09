from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from pyspark.sql import SparkSession

from queries.common_utils import (
    check_query_result_pd,
    get_table_path,
    run_query_generic,
)
from settings import Settings

if TYPE_CHECKING:
    from pyspark.sql import DataFrame

settings = Settings()

# Spark 4.x runs on Java 17 or 21 (prefer higher version)
SUPPORTED_JAVA_VERSIONS = (21, 17)


def _ensure_java_home() -> None:
    """Point JAVA_HOME at a Spark-compatible JDK (if not already set)."""
    if os.environ.get("JAVA_HOME"):
        return

    # Note: if nothing suitable is found, Spark's own resolution still applies
    if sys.platform == "darwin":
        candidates = []
        for version in SUPPORTED_JAVA_VERSIONS:
            try:
                home = subprocess.run(
                    ["/usr/libexec/java_home", "-v", str(version)],
                    capture_output=True,
                    text=True,
                    check=True,
                ).stdout.strip()
            except (OSError, subprocess.CalledProcessError):
                home = ""
            if home:
                candidates.append(home)

            candidates.append(f"/opt/homebrew/opt/openjdk@{version}")
            candidates.append(f"/usr/local/opt/openjdk@{version}")

        for home in candidates:
            if (Path(home) / "bin" / "java").exists():
                os.environ["JAVA_HOME"] = home
                return

    # TODO: automatically determine a suitable JAVA_HOME on other platforms
    # if sys.platform == ...:
    #     ...


def get_or_create_spark() -> SparkSession:
    _ensure_java_home()

    spark = (
        SparkSession.builder.appName("spark_queries")
        .master("local[*]")
        # Bind the local driver to loopback. Without this, Spark resolves the
        # machine hostname, which on many setups (e.g. behind a VPN) maps to a
        # non-loopback address the block manager then cannot reach, causing
        # "TaskResultLost" failures.
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.memory", settings.run.spark_driver_memory)
        .config("spark.executor.memory", settings.run.spark_executor_memory)
        .config("spark.log.level", settings.run.spark_log_level)
        .getOrCreate()
    )
    return spark


def _read_ds(table_name: str) -> DataFrame:
    if settings.run.io_type == "skip":
        # TODO: Persist data in memory before query
        msg = "cannot run PySpark starting from an in-memory representation"
        raise RuntimeError(msg)

    path = get_table_path(table_name)

    if settings.run.io_type == "parquet":
        df = get_or_create_spark().read.parquet(str(path))
    elif settings.run.io_type == "csv":
        df = get_or_create_spark().read.csv(str(path), header=True, inferSchema=True)
    else:
        msg = f"unsupported file type: {settings.run.io_type!r}"
        raise ValueError(msg)

    df.createOrReplaceTempView(table_name)
    return df


def get_line_item_ds() -> DataFrame:
    return _read_ds("lineitem")


def get_orders_ds() -> DataFrame:
    return _read_ds("orders")


def get_customer_ds() -> DataFrame:
    return _read_ds("customer")


def get_region_ds() -> DataFrame:
    return _read_ds("region")


def get_nation_ds() -> DataFrame:
    return _read_ds("nation")


def get_supplier_ds() -> DataFrame:
    return _read_ds("supplier")


def get_part_ds() -> DataFrame:
    return _read_ds("part")


def get_part_supp_ds() -> DataFrame:
    return _read_ds("partsupp")


def run_query(query_number: int, df: DataFrame) -> None:
    query = df.toPandas
    run_query_generic(
        query, query_number, "pyspark", query_checker=check_query_result_pd
    )
