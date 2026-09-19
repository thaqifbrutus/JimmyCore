import json

from sqlalchemy.orm import Session

from app.models.report import QualityReport
from app.models.audit_log import AuditLog
from app.services.profiler import determine_overall_status


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
    Creates and commits a QualityReport plus its AuditLog entry. Shared by
    the upload flow (reports.py's trigger_profile) and the
    government-catalog flow (catalog.py's analyze endpoint) — exactly one
    of dataset_id / catalog_dataset_id should be passed, matching the
    ck_report_exactly_one_source constraint on QualityReport. Extracted
    here specifically so both flows build a report identically rather
    than maintaining two near-copies of the same commit-then-log sequence.
    """
    overall_status = determine_overall_status(profile["issues"])

    report = QualityReport(
        dataset_id=dataset_id,
        catalog_dataset_id=catalog_dataset_id,
        profile_data=profile,
        ai_summary=json.dumps(ai_summary),
        overall_status=overall_status,
    )
    db.add(report)
    db.commit()
    db.refresh(report)

    log = AuditLog(
        dataset_id=dataset_id,
        report_id=report.id,
        action=audit_action,
        detail=(
            f"Profile and AI summary generated. "
            f"Status: {overall_status}. "
            f"Issues found: {len(profile['issues'])}. "
            f"AI summary status: {ai_summary.get('status')}"
        ),
    )
    db.add(log)
    db.commit()

    return report
