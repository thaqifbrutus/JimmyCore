"""
Loads the actual DataFrame behind a QualityReport, regardless of which
flow created it (uploaded CSV or government-catalog dataset).

This is the coupling layer — it imports QualityReport/Dataset/CatalogDataset
so nothing downstream has to. Everything downstream of
load_dataframe_for_report() sees only a DataFrame.
"""
import os

import pandas as pd
from sqlalchemy.orm import Session

from app.models.catalog_dataset import CatalogDataset
from app.models.dataset import Dataset
from app.models.report import QualityReport
from app.services import gov_data_client


UPLOAD_DIR = "file_uploads"


class DataLoadError(Exception):
    """Raised when the data behind a report cannot be loaded."""
    pass


def load_dataframe_for_report(report: QualityReport, db: Session) -> pd.DataFrame:
    """
    Exactly one of report.dataset_id / report.catalog_dataset_id will be
    set, per the ck_report_exactly_one_source constraint.
    """
    if report.dataset_id is not None:
        dataset = db.query(Dataset).filter(Dataset.id == report.dataset_id).first()
        if dataset is None:
            raise DataLoadError(
                f"Dataset {report.dataset_id} referenced by report "
                f"{report.id} was not found."
            )
        file_path = os.path.join(UPLOAD_DIR, dataset.filename)
        if not os.path.exists(file_path):
            raise DataLoadError(
                f"Uploaded file for dataset {dataset.id} is missing on "
                f"disk: {file_path}"
            )
        return pd.read_csv(file_path)

    if report.catalog_dataset_id is not None:
        try:
            return gov_data_client.get_dataset_dataframe(
                db, report.catalog_dataset_id
            )
        except gov_data_client.GovAPIError as e:
            raise DataLoadError(
                f"Could not load government dataset "
                f"'{report.catalog_dataset_id}': {e}"
            )

    raise DataLoadError(
        "Report has neither dataset_id nor catalog_dataset_id — should be "
        "impossible given ck_report_exactly_one_source."
    )