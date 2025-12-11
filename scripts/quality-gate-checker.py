#!/usr/bin/env python3
"""
Quality Gate Checker

Validates quality gates for multi-phase development workflow.
Supports Analysis, Implementation, and Validation phases with configurable thresholds.

Usage:
    python3 quality-gate-checker.py --phase analysis --config .quality-gates.json
    python3 quality-gate-checker.py --phase implementation --coverage-report coverage/lcov.info
    python3 quality-gate-checker.py --phase validation --exit-code
    python3 quality-gate-checker.py --phase all --strict
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class Phase(Enum):
    """Development workflow phases"""
    ANALYSIS = "analysis"
    IMPLEMENTATION = "implementation"
    VALIDATION = "validation"
    ALL = "all"


class GateStatus(Enum):
    """Quality gate pass/fail status"""
    PASS = "pass"
    FAIL = "fail"
    SKIP = "skip"


@dataclass
class GateResult:
    """Result of quality gate check"""
    phase: str
    score: float
    threshold: float
    status: GateStatus
    criteria: Dict[str, Any]
    details: str


class QualityGateChecker:
    """Validates quality gates across development phases"""

    DEFAULT_THRESHOLDS = {
        "analysis": 95,
        "implementation": 80,
        "validation": 85
    }

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize quality gate checker

        Args:
            config_path: Path to .quality-gates.json configuration file
        """
        self.config = self._load_config(config_path) if config_path else {}
        self.thresholds = self.config.get("thresholds", self.DEFAULT_THRESHOLDS)

    def _load_config(self, config_path: Path) -> Dict[str, Any]:
        """Load quality gate configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"⚠️  Warning: Config file not found: {config_path}", file=sys.stderr)
            print(f"⚠️  Using default thresholds: {self.DEFAULT_THRESHOLDS}", file=sys.stderr)
            return {}
        except json.JSONDecodeError as e:
            print(f"❌ Error: Invalid JSON in config file: {e}", file=sys.stderr)
            sys.exit(1)

    def check_phase1_analysis(self) -> GateResult:
        """
        Check Phase 1: Analysis quality gate (95% threshold)

        Criteria:
        - requirements_complete: boolean
        - domain_model_validated: boolean
        - architecture_documented: boolean
        - risks_identified: boolean
        """
        phase_config = self.config.get("phase1", {})
        threshold = self.thresholds.get("analysis", 95)

        criteria = {
            "requirements_complete": phase_config.get("requirements_complete", False),
            "domain_model_validated": phase_config.get("domain_model_validated", False),
            "architecture_documented": phase_config.get("architecture_documented", False),
            "risks_identified": phase_config.get("risks_identified", False)
        }

        # Calculate score: each criterion worth 25%
        score = sum(1 for v in criteria.values() if v) / len(criteria) * 100

        status = GateStatus.PASS if score >= threshold else GateStatus.FAIL

        details = self._format_analysis_details(criteria, score, threshold)

        return GateResult(
            phase="analysis",
            score=score,
            threshold=threshold,
            status=status,
            criteria=criteria,
            details=details
        )

    def check_phase2_implementation(
        self,
        coverage_report: Optional[Path] = None,
        lint_report: Optional[Path] = None
    ) -> GateResult:
        """
        Check Phase 2: Implementation quality gate (80% threshold)

        Criteria:
        - test_coverage: numeric >= 80
        - security_scan: pass/fail
        - lint_errors: numeric == 0
        - code_review: pass/fail
        """
        phase_config = self.config.get("phase2", {})
        threshold = self.thresholds.get("implementation", 80)

        # Parse coverage report if provided
        test_coverage = self._parse_coverage(coverage_report) if coverage_report else phase_config.get("test_coverage", 0)

        # Parse lint report if provided
        lint_errors = self._parse_lint(lint_report) if lint_report else phase_config.get("lint_errors", 0)

        criteria = {
            "test_coverage": test_coverage,
            "security_scan": phase_config.get("security_scan", "fail"),
            "lint_errors": lint_errors,
            "code_review": phase_config.get("code_review", "fail")
        }

        # Calculate score: each criterion worth 25%
        coverage_score = 25 if test_coverage >= 80 else 0
        security_score = 25 if criteria["security_scan"] == "pass" else 0
        lint_score = 25 if criteria["lint_errors"] == 0 else 0
        review_score = 25 if criteria["code_review"] == "pass" else 0

        score = coverage_score + security_score + lint_score + review_score

        status = GateStatus.PASS if score >= threshold else GateStatus.FAIL

        details = self._format_implementation_details(criteria, score, threshold)

        return GateResult(
            phase="implementation",
            score=score,
            threshold=threshold,
            status=status,
            criteria=criteria,
            details=details
        )

    def check_phase3_validation(
        self,
        integration_report: Optional[Path] = None,
        perf_report: Optional[Path] = None
    ) -> GateResult:
        """
        Check Phase 3: Validation quality gate (85% threshold)

        Criteria:
        - integration_tests: pass/fail
        - performance_benchmarks: pass/fail
        - documentation_complete: boolean
        - acceptance_criteria_met: boolean
        """
        phase_config = self.config.get("phase3", {})
        threshold = self.thresholds.get("validation", 85)

        # Parse integration test results if provided
        integration_tests = self._parse_integration_tests(integration_report) if integration_report else phase_config.get("integration_tests", "fail")

        # Parse performance benchmarks if provided
        performance_benchmarks = self._parse_performance(perf_report) if perf_report else phase_config.get("performance_benchmarks", "fail")

        criteria = {
            "integration_tests": integration_tests,
            "performance_benchmarks": performance_benchmarks,
            "documentation_complete": phase_config.get("documentation_complete", False),
            "acceptance_criteria_met": phase_config.get("acceptance_criteria_met", False)
        }

        # Calculate score: each criterion worth 25%
        integration_score = 25 if criteria["integration_tests"] == "pass" else 0
        performance_score = 25 if criteria["performance_benchmarks"] == "pass" else 0
        documentation_score = 25 if criteria["documentation_complete"] else 0
        acceptance_score = 25 if criteria["acceptance_criteria_met"] else 0

        score = integration_score + performance_score + documentation_score + acceptance_score

        status = GateStatus.PASS if score >= threshold else GateStatus.FAIL

        details = self._format_validation_details(criteria, score, threshold)

        return GateResult(
            phase="validation",
            score=score,
            threshold=threshold,
            status=status,
            criteria=criteria,
            details=details
        )

    def _parse_coverage(self, coverage_report: Path) -> float:
        """Parse coverage from lcov.info or coverage.json"""
        try:
            # Try JSON format first (jest, pytest-cov with json reporter)
            if coverage_report.suffix == '.json':
                with open(coverage_report, 'r') as f:
                    data = json.load(f)
                    # Jest format
                    if 'total' in data:
                        return data['total'].get('lines', {}).get('pct', 0)
                    # pytest-cov format
                    elif 'totals' in data:
                        return data['totals'].get('percent_covered', 0)

            # Try lcov format (lcov.info)
            elif coverage_report.suffix == '.info' or 'lcov' in coverage_report.name:
                with open(coverage_report, 'r') as f:
                    lines_hit = 0
                    lines_found = 0
                    for line in f:
                        if line.startswith('LH:'):
                            lines_hit += int(line.split(':')[1])
                        elif line.startswith('LF:'):
                            lines_found += int(line.split(':')[1])
                    return (lines_hit / lines_found * 100) if lines_found > 0 else 0

            return 0
        except Exception as e:
            print(f"⚠️  Warning: Could not parse coverage report: {e}", file=sys.stderr)
            return 0

    def _parse_lint(self, lint_report: Path) -> int:
        """Parse lint errors from JSON report"""
        try:
            with open(lint_report, 'r') as f:
                data = json.load(f)

                # ESLint format
                if isinstance(data, list):
                    return sum(len(file.get('messages', [])) for file in data
                               if any(m.get('severity', 0) == 2 for m in file.get('messages', [])))

                # Generic format
                return data.get('errorCount', 0)
        except Exception as e:
            print(f"⚠️  Warning: Could not parse lint report: {e}", file=sys.stderr)
            return 0

    def _parse_integration_tests(self, integration_report: Path) -> str:
        """Parse integration test results from JSON report"""
        try:
            with open(integration_report, 'r') as f:
                data = json.load(f)

                # Jest/pytest format
                if 'success' in data:
                    return "pass" if data['success'] else "fail"

                # Generic format
                total = data.get('numTotalTests', data.get('total', 0))
                passed = data.get('numPassedTests', data.get('passed', 0))

                return "pass" if total > 0 and total == passed else "fail"
        except Exception as e:
            print(f"⚠️  Warning: Could not parse integration test report: {e}", file=sys.stderr)
            return "fail"

    def _parse_performance(self, perf_report: Path) -> str:
        """Parse performance benchmarks from JSON report"""
        try:
            with open(perf_report, 'r') as f:
                data = json.load(f)

                # k6 format
                if 'metrics' in data:
                    metrics = data['metrics']
                    http_req_duration = metrics.get('http_req_duration', {})
                    p95 = http_req_duration.get('values', {}).get('p(95)', float('inf'))
                    return "pass" if p95 < 200 else "fail"

                # Generic format
                return "pass" if data.get('passed', False) else "fail"
        except Exception as e:
            print(f"⚠️  Warning: Could not parse performance report: {e}", file=sys.stderr)
            return "fail"

    def _format_analysis_details(self, criteria: Dict[str, bool], score: float, threshold: float) -> str:
        """Format analysis phase details"""
        status_icon = "✅" if score >= threshold else "❌"

        details = f"\n{'='*60}\n"
        details += f"Phase 1: Analysis Quality Gate\n"
        details += f"{'='*60}\n\n"

        for key, value in criteria.items():
            icon = "✅" if value else "❌"
            label = key.replace('_', ' ').title()
            details += f"{icon} {label}: {'PASS' if value else 'FAIL'}\n"

        details += f"\n{'-'*60}\n"
        details += f"Score: {score:.0f}% (threshold: {threshold:.0f}%)\n"
        details += f"Status: {status_icon} {'PASSED' if score >= threshold else 'FAILED'}\n"
        details += f"{'-'*60}\n"

        return details

    def _format_implementation_details(self, criteria: Dict[str, Any], score: float, threshold: float) -> str:
        """Format implementation phase details"""
        status_icon = "✅" if score >= threshold else "❌"

        details = f"\n{'='*60}\n"
        details += f"Phase 2: Implementation Quality Gate\n"
        details += f"{'='*60}\n\n"

        # Test coverage
        coverage = criteria["test_coverage"]
        coverage_icon = "✅" if coverage >= 80 else "❌"
        details += f"{coverage_icon} Test Coverage: {coverage:.1f}% (threshold: 80%)\n"

        # Security scan
        security = criteria["security_scan"]
        security_icon = "✅" if security == "pass" else "❌"
        details += f"{security_icon} Security Scan: {security.upper()}\n"

        # Lint errors
        lint = criteria["lint_errors"]
        lint_icon = "✅" if lint == 0 else "❌"
        details += f"{lint_icon} Lint Errors: {lint} (threshold: 0)\n"

        # Code review
        review = criteria["code_review"]
        review_icon = "✅" if review == "pass" else "❌"
        details += f"{review_icon} Code Review: {review.upper()}\n"

        details += f"\n{'-'*60}\n"
        details += f"Score: {score:.0f}% (threshold: {threshold:.0f}%)\n"
        details += f"Status: {status_icon} {'PASSED' if score >= threshold else 'FAILED'}\n"
        details += f"{'-'*60}\n"

        return details

    def _format_validation_details(self, criteria: Dict[str, Any], score: float, threshold: float) -> str:
        """Format validation phase details"""
        status_icon = "✅" if score >= threshold else "❌"

        details = f"\n{'='*60}\n"
        details += f"Phase 3: Validation Quality Gate\n"
        details += f"{'='*60}\n\n"

        # Integration tests
        integration = criteria["integration_tests"]
        integration_icon = "✅" if integration == "pass" else "❌"
        details += f"{integration_icon} Integration Tests: {integration.upper()}\n"

        # Performance benchmarks
        performance = criteria["performance_benchmarks"]
        performance_icon = "✅" if performance == "pass" else "❌"
        details += f"{performance_icon} Performance Benchmarks: {performance.upper()}\n"

        # Documentation
        docs = criteria["documentation_complete"]
        docs_icon = "✅" if docs else "❌"
        details += f"{docs_icon} Documentation Complete: {'YES' if docs else 'NO'}\n"

        # Acceptance criteria
        acceptance = criteria["acceptance_criteria_met"]
        acceptance_icon = "✅" if acceptance else "❌"
        details += f"{acceptance_icon} Acceptance Criteria Met: {'YES' if acceptance else 'NO'}\n"

        details += f"\n{'-'*60}\n"
        details += f"Score: {score:.0f}% (threshold: {threshold:.0f}%)\n"
        details += f"Status: {status_icon} {'PASSED' if score >= threshold else 'FAILED'}\n"
        details += f"{'-'*60}\n"

        return details

    def check_all_phases(self, **kwargs) -> Dict[str, GateResult]:
        """Check all phases and return combined results"""
        return {
            "analysis": self.check_phase1_analysis(),
            "implementation": self.check_phase2_implementation(
                coverage_report=kwargs.get("coverage_report"),
                lint_report=kwargs.get("lint_report")
            ),
            "validation": self.check_phase3_validation(
                integration_report=kwargs.get("integration_report"),
                perf_report=kwargs.get("perf_report")
            )
        }

    def export_json(self, results: Dict[str, GateResult], output_path: Path) -> None:
        """Export results to JSON file"""
        data = {
            phase: {
                "score": result.score,
                "threshold": result.threshold,
                "status": result.status.value,
                "criteria": result.criteria
            }
            for phase, result in results.items()
        }

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"\n📄 Results exported to: {output_path}")


def main():
    """Main entry point for CLI"""
    parser = argparse.ArgumentParser(
        description="Quality gate checker for multi-phase development workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check analysis phase
  python3 quality-gate-checker.py --phase analysis --config .quality-gates.json

  # Check implementation phase with coverage report
  python3 quality-gate-checker.py --phase implementation \\
    --config .quality-gates.json \\
    --coverage-report coverage/lcov.info \\
    --lint-report lint-results.json

  # Check validation phase
  python3 quality-gate-checker.py --phase validation \\
    --config .quality-gates.json \\
    --integration-report test-results.json \\
    --perf-report benchmarks/k6-results.json

  # Check all phases and exit with status code
  python3 quality-gate-checker.py --phase all --config .quality-gates.json --exit-code

  # Export results to JSON
  python3 quality-gate-checker.py --phase all --output quality-report.json
        """
    )

    parser.add_argument(
        "--phase",
        type=str,
        choices=["analysis", "implementation", "validation", "all"],
        required=True,
        help="Phase to validate"
    )

    parser.add_argument(
        "--config",
        type=Path,
        help="Path to .quality-gates.json configuration file"
    )

    parser.add_argument(
        "--coverage-report",
        type=Path,
        help="Path to coverage report (lcov.info or coverage.json)"
    )

    parser.add_argument(
        "--lint-report",
        type=Path,
        help="Path to lint report (JSON format)"
    )

    parser.add_argument(
        "--integration-report",
        type=Path,
        help="Path to integration test results (JSON format)"
    )

    parser.add_argument(
        "--perf-report",
        type=Path,
        help="Path to performance benchmarks (JSON format)"
    )

    parser.add_argument(
        "--output",
        type=Path,
        help="Output path for JSON report"
    )

    parser.add_argument(
        "--exit-code",
        action="store_true",
        help="Exit with non-zero code if gate fails"
    )

    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any phase fails (only with --phase all)"
    )

    args = parser.parse_args()

    # Initialize checker
    checker = QualityGateChecker(config_path=args.config)

    # Check specified phase(s)
    if args.phase == "all":
        results = checker.check_all_phases(
            coverage_report=args.coverage_report,
            lint_report=args.lint_report,
            integration_report=args.integration_report,
            perf_report=args.perf_report
        )

        # Print all results
        for phase, result in results.items():
            print(result.details)

        # Export if requested
        if args.output:
            checker.export_json(results, args.output)

        # Determine exit code
        if args.exit_code or args.strict:
            failed_phases = [phase for phase, result in results.items()
                           if result.status == GateStatus.FAIL]
            if failed_phases:
                print(f"\n❌ Failed phases: {', '.join(failed_phases)}")
                sys.exit(1)
            else:
                print("\n✅ All phases passed!")
                sys.exit(0)

    else:
        # Check single phase
        if args.phase == "analysis":
            result = checker.check_phase1_analysis()
        elif args.phase == "implementation":
            result = checker.check_phase2_implementation(
                coverage_report=args.coverage_report,
                lint_report=args.lint_report
            )
        elif args.phase == "validation":
            result = checker.check_phase3_validation(
                integration_report=args.integration_report,
                perf_report=args.perf_report
            )

        # Print result
        print(result.details)

        # Export if requested
        if args.output:
            checker.export_json({args.phase: result}, args.output)

        # Exit with status code if requested
        if args.exit_code:
            sys.exit(0 if result.status == GateStatus.PASS else 1)


if __name__ == "__main__":
    main()
