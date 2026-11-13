#!/usr/bin/env python3
import importlib.util
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TERMS = ROOT / "site" / "terms"


def load_site_terms():
    spec = importlib.util.spec_from_file_location("site_terms", ROOT / "tools" / "site_terms.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def clause(text, number):
    found = re.search(rf"^{re.escape(number)}\. .*$", text, re.M)
    return found.group(0) if found else ""


def row(page, start):
    found = re.search(rf"<tr><td>{re.escape(start)}.*?</tr>", page)
    return found.group(0) if found else ""


class AgreementTest(unittest.TestCase):
    def setUp(self):
        self.en = (TERMS / "agreement-en.md").read_text(encoding="utf-8")
        self.ko = (TERMS / "agreement-ko.md").read_text(encoding="utf-8")

    def test_agreement_incorporates_the_dpa(self):
        self.assertIn("Data Processing Agreement", clause(self.en, "1.3"))
        self.assertIn("forms part of this Agreement", clause(self.en, "1.3"))
        self.assertIn("데이터 처리 계약", clause(self.ko, "1.3"))
        self.assertIn("일부를 이룬다", clause(self.ko, "1.3"))
        self.assertIn("the DPA", clause(self.en, "11.1"))
        self.assertIn("DPA", clause(self.ko, "11.1"))
        self.assertIn("as the DPA states", clause(self.en, "6.4"))
        self.assertIn("DPA에 따라", clause(self.ko, "6.4"))

    def test_dpa_cancellation_is_an_exception_to_no_cancellation(self):
        for number in ("10.3", "14.3"):
            self.assertIn("clause 5.2 of the DPA", clause(self.en, number), number)
            self.assertIn("DPA 제5.2조", clause(self.ko, number), number)

    def test_agreement_names_no_url(self):
        for name, text in (("en", self.en), ("ko", self.ko)):
            self.assertIsNone(re.search(r"https?://|sparkpipe\.ai/|\.html\b|\.txt\b", text), name)

    def test_fleets_carry_no_ai_software_of_ours(self):
        self.assertIn("we run no AI model for Customer", clause(self.en, "5.2"))
        self.assertIn("공급자는 고객을 위하여 어떠한 인공지능 모델도 실행하지 아니한다", clause(self.ko, "5.2"))
        self.assertIn("with no AI model and no software whose purpose is to run one", clause(self.en, "7.1"))
        self.assertIn("인공지능 모델이나 이를 실행하는 것을 목적으로 하는 소프트웨어는 두지 아니한다", clause(self.ko, "7.1"))
        for text in (self.en, self.ko):
            self.assertNotIn("Preinstalled", text)
            self.assertNotIn("preinstall", text)
            self.assertNotIn("사전 설치", text)
            self.assertNotIn("미리 설치", text)

    def test_dpa_names_no_module_licensor(self):
        en = (TERMS / "dpa-en.md").read_text(encoding="utf-8")
        ko = (TERMS / "dpa-ko.md").read_text(encoding="utf-8")
        self.assertNotIn("licensor", clause(en, "5.1"))
        self.assertNotIn("라이선스 제공자", clause(ko, "5.1"))
        self.assertNotIn("licensed from others", self.en)
        self.assertNotIn("타인으로부터 라이선스", self.ko)


class VersionTest(unittest.TestCase):
    def test_version_line_is_part_of_the_hashed_text(self):
        site_terms = load_site_terms()
        for page in site_terms.PAGES:
            self.assertNotIn("version", page)
            lines = (TERMS / page["src"]).read_text(encoding="utf-8").splitlines()
            rendered = site_terms.page(page)
            self.assertIn(f'<p class="effective">{site_terms.inline(lines[2].strip())}</p>', rendered, page["src"])
        twins = {p["src"]: (TERMS / p["src"]).read_text(encoding="utf-8").splitlines()[2] for p in site_terms.PAGES}
        self.assertTrue(re.fullmatch(r"Version of \d{1,2} \w+ \d{4}", twins["agreement-en.md"]))
        self.assertTrue(re.fullmatch(r"\d{4}년 \d{1,2}월 \d{1,2}일 판", twins["agreement-ko.md"]))

    def test_source_without_version_is_refused(self):
        site_terms = load_site_terms()
        with self.assertRaises(ValueError):
            site_terms.body("# Title\n\n## 1. Heading\n\n1.1. Text.\n")


class PrivacyTest(unittest.TestCase):
    def setUp(self):
        self.en = (ROOT / "site" / "privacy.html").read_text(encoding="utf-8")
        self.ko = (ROOT / "site" / "privacy-ko.html").read_text(encoding="utf-8")

    def test_request_records_and_their_deletion_are_in_the_privacy_policy(self):
        en = row(self.en, "Operational records for each request")
        ko = row(self.ko, "요청별 운영기록")
        self.assertIn("no prompt or reply text and no token numbers", en)
        self.assertNotIn("hash of the prompt", en)
        self.assertNotIn("rebuilt", en)
        self.assertIn("within 30 days", en)
        self.assertIn("토큰 번호는 포함하지 않음", ko)
        self.assertNotIn("단방향 해시", ko)
        self.assertNotIn("복원", ko)
        self.assertIn("30일 이내에 삭제", ko)
        for name in ("dpa-en.md", "dpa-ko.md"):
            self.assertEqual(clause((TERMS / name).read_text(encoding="utf-8"), "9.3"), "", name)

    def test_members_invitations_and_audit_log_are_disclosed(self):
        self.assertTrue(row(self.en, "Account members and invitations"))
        self.assertTrue(row(self.en, "Account audit log"))
        self.assertTrue(row(self.ko, "계정 구성원 및 초대"))
        self.assertTrue(row(self.ko, "계정 감사기록"))
        self.assertIn("1 year", row(self.en, "Account audit log"))
        self.assertIn("1년", row(self.ko, "계정 감사기록"))
        self.assertIn("a member's email address only in the lines that invite or add that member", self.en)
        self.assertIn("구성원의 이메일 주소는 그 구성원을 초대하거나 추가하는 줄에만", self.ko)

    def test_acceptance_records_cover_licenses_from_the_catalogue(self):
        en = row(self.en, "Acceptance records")
        ko = row(self.ko, "동의기록")
        self.assertIn("the license of software chosen from the Catalogue", en)
        self.assertIn("chooses from the Catalogue what to install on a fleet", en)
        self.assertIn("소프트웨어 목록에서 고른 소프트웨어의 라이선스", ko)
        self.assertIn("플릿에 설치할 것을 소프트웨어 목록에서 고를 때", ko)
        self.assertIn("the hashes of the texts accepted", en)
        self.assertIn("동의한 문서의 해시", ko)
        self.assertIn("the member who made a choice from the Catalogue", en)
        self.assertIn("소프트웨어 목록에서 고른 구성원", ko)
        for text in (en, ko):
            self.assertIn("IP", text)
        self.assertIn("its choices from the Catalogue", self.en)
        self.assertIn("소프트웨어 목록에서의 선택을 기록하는 줄", self.ko)

    def test_processing_location_has_no_fixed_machine_count(self):
        self.assertNotIn("sixteen", self.en)
        self.assertNotIn("16대", self.ko)
        self.assertIn('href="fleets.html"', self.en)
        self.assertIn('href="fleets-ko.html"', self.ko)


if __name__ == "__main__":
    sys.exit(0 if unittest.main(exit=False).result.wasSuccessful() else 1)
