import json

from sqlalchemy.orm import Session

from app.models.report import QualityReport
from app.models.audit_log import AuditLog


def persist_report(
    db: Session,
    profile: dict,
    ai_summary: dict,
    *,
    dataset_id=None,
    catalog_dataset_id=None,
    audit_action: str,
) -> QualityReport:
    """
    Creates and commits a QualityReport plus its AuditLog entry.

    Exactly one of dataset_id / catalog_dataset_id should be passed,
    matching the ck_report_exactly_one_source constraint. The
    overall_status column is left at its default — the "quality verdict"
    concept was dropped when the app moved to a chat-based analyst model;
    the column stays for schema compatibility.
    """
    report = QualityReport(
        dataset_id=dataset_id,
        catalog_dataset_id=catalog_dataset_id,
        profile_data=profile,
        ai_summary=json.dumps(ai_summary),
    )
    db.add(report)
    db.commit()
    db.refresh(report)

    log = AuditLog(
        dataset_id=dataset_id,
        report_id=report.id,
        action=audit_action,
        detail=(
            f"Profile and AI overview generated. "
            f"Issues found: {len(profile.get('issues', []))}. "
            f"AI overview status: {ai_summary.get('status')}"
        ),
    )
    db.add(log)
    db.commit()

    return report