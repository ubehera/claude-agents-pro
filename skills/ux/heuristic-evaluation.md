---
name: heuristic-evaluation
description: Load when user needs heuristic evaluation, expert review, Nielsen's 10 usability heuristics, cognitive walkthrough, or UX audit methodology. Covers expert inspection methods.
trigger_keywords: [heuristic evaluation, nielsen heuristics, usability heuristics, expert review, ux audit, cognitive walkthrough, severity rating, usability inspection, design critique, ux review]
---

# Heuristic Evaluation Skill

Expert inspection methods for identifying usability issues without user testing.

## Core Concepts

- **Nielsen's 10 Heuristics**: Industry-standard usability principles (visibility, consistency, error prevention, etc.) providing systematic framework for expert evaluation
- **Severity Rating Scale**: 0-4 scale (Cosmetic to Catastrophic) combining frequency, impact, and persistence factors to prioritize findings for remediation
- **Multiple Evaluator Approach**: 3-5 independent evaluators find 75%+ of issues; single evaluator misses over half - aggregation essential for comprehensive coverage
- **Cognitive Walkthrough**: Step-by-step task analysis evaluating learnability through four questions about goal visibility, action discoverability, association, and feedback
- **Actionable Recommendations**: Each finding must include specific location, heuristic violated, clear description, and concrete solution - vague findings cannot be fixed

## Nielsen's 10 Usability Heuristics

### Complete Heuristics Reference

```python
from dataclasses import dataclass
from typing import List, Dict
from enum import Enum

class SeverityRating(Enum):
    COSMETIC = 0  # "I don't agree this is a problem"
    MINOR = 1  # "Cosmetic problem - fix if time"
    MODERATE = 2  # "Minor usability problem - low priority"
    MAJOR = 3  # "Major usability problem - high priority"
    CATASTROPHIC = 4  # "Usability catastrophe - must fix"

@dataclass
class Heuristic:
    """Nielsen Norman Group Usability Heuristic"""

    number: int
    name: str
    description: str
    key_questions: List[str]
    positive_examples: List[str]
    negative_examples: List[str]
    evaluation_tips: List[str]

NIELSEN_HEURISTICS = {
    1: Heuristic(
        number=1,
        name="Visibility of System Status",
        description="The design should always keep users informed about what is going on, through appropriate feedback within a reasonable amount of time.",
        key_questions=[
            "Does the user know where they are in the system?",
            "Is there feedback for user actions?",
            "Are progress indicators shown for long operations?",
            "Is the current state clearly visible?"
        ],
        positive_examples=[
            "Progress bars during file uploads",
            "Highlighted current page in navigation",
            "Confirmation messages after form submission",
            "Loading spinners during data fetch",
            "Breadcrumbs showing location in hierarchy"
        ],
        negative_examples=[
            "No feedback after clicking a button",
            "Page loads with no loading indicator",
            "User doesn't know if action succeeded",
            "No indication of current step in a process"
        ],
        evaluation_tips=[
            "Click every interactive element - is there feedback?",
            "Try slow network - are loading states present?",
            "Check if current location is always clear"
        ]
    ),
    2: Heuristic(
        number=2,
        name="Match Between System and Real World",
        description="The design should speak the users' language. Use words, phrases, and concepts familiar to the user, rather than internal jargon.",
        key_questions=[
            "Is the language user-friendly, not technical?",
            "Do icons and metaphors make sense?",
            "Is information in logical, natural order?",
            "Are conventions from the real world followed?"
        ],
        positive_examples=[
            "Shopping cart icon for e-commerce",
            "Folder/file metaphor for organization",
            "Calendar showing days in expected order",
            "Using 'Sign up' instead of 'Create account instance'"
        ],
        negative_examples=[
            "Error: 'HTTP 500 Internal Server Error'",
            "Technical jargon in user-facing copy",
            "Unexpected icon meanings",
            "Form fields in illogical order"
        ],
        evaluation_tips=[
            "Read all copy as if you were a new user",
            "Check if icons are universally understood",
            "Verify terminology matches user expectations"
        ]
    ),
    3: Heuristic(
        number=3,
        name="User Control and Freedom",
        description="Users often perform actions by mistake. They need a clearly marked 'emergency exit' to leave the unwanted action without having to go through an extended process.",
        key_questions=[
            "Can users undo/redo actions?",
            "Is there a clear way to cancel or go back?",
            "Can users exit processes mid-way?",
            "Are there confirmation dialogs for destructive actions?"
        ],
        positive_examples=[
            "Undo button after deleting email",
            "Cancel button in forms and modals",
            "Browser back button works as expected",
            "'Are you sure?' for irreversible actions",
            "Exit button in multi-step processes"
        ],
        negative_examples=[
            "No way to undo a delete",
            "Can't go back in a wizard",
            "Forced to complete a process once started",
            "No cancel option in modals"
        ],
        evaluation_tips=[
            "Try to exit every process mid-way",
            "Look for undo options after actions",
            "Check if back button works everywhere"
        ]
    ),
    4: Heuristic(
        number=4,
        name="Consistency and Standards",
        description="Users should not have to wonder whether different words, situations, or actions mean the same thing. Follow platform and industry conventions.",
        key_questions=[
            "Are UI elements consistent throughout?",
            "Do similar actions behave the same way?",
            "Are platform conventions followed?",
            "Is terminology consistent?"
        ],
        positive_examples=[
            "Primary buttons look the same everywhere",
            "Delete always uses red color",
            "Links are consistently underlined",
            "Same icon always means same action"
        ],
        negative_examples=[
            "Submit button looks different on different pages",
            "Different terms for same concept (Save vs Submit vs Apply)",
            "Inconsistent date formats",
            "Different navigation patterns on different pages"
        ],
        evaluation_tips=[
            "Compare similar screens side by side",
            "Check if same words/icons are used consistently",
            "Verify platform conventions are followed"
        ]
    ),
    5: Heuristic(
        number=5,
        name="Error Prevention",
        description="Good design prevents problems from occurring in the first place. Either eliminate error-prone conditions, or check for them and present users with a confirmation option.",
        key_questions=[
            "Are constraints in place to prevent errors?",
            "Are confirmation steps before destructive actions?",
            "Is input validated in real-time?",
            "Are helpful defaults provided?"
        ],
        positive_examples=[
            "Date picker instead of free text input",
            "Disable submit until form is valid",
            "Confirmation before delete",
            "Auto-complete for known values",
            "Inline validation as user types"
        ],
        negative_examples=[
            "Free text where dropdown would work",
            "No validation until form submission",
            "Immediate delete without confirmation",
            "No constraints on numeric inputs"
        ],
        evaluation_tips=[
            "Try to enter invalid data everywhere",
            "Look for places where errors could be prevented",
            "Check if destructive actions require confirmation"
        ]
    ),
    6: Heuristic(
        number=6,
        name="Recognition Rather Than Recall",
        description="Minimize the user's memory load by making elements, actions, and options visible. The user should not have to remember information from one part to another.",
        key_questions=[
            "Are instructions visible when needed?",
            "Are options visible rather than hidden?",
            "Is recently used information available?",
            "Can users recognize rather than recall?"
        ],
        positive_examples=[
            "Recently viewed items section",
            "Autocomplete with suggestions",
            "Visible navigation menu",
            "In-context help and tooltips",
            "Search history visible"
        ],
        negative_examples=[
            "Hidden navigation (hamburger overuse)",
            "Must remember codes or IDs",
            "No hints for format requirements",
            "Important info hidden in previous steps"
        ],
        evaluation_tips=[
            "Check if users need to remember things",
            "Look for hidden information that should be visible",
            "Verify help is available in context"
        ]
    ),
    7: Heuristic(
        number=7,
        name="Flexibility and Efficiency of Use",
        description="Shortcuts — hidden from novice users — can speed up interaction for experts. Allow users to tailor frequent actions.",
        key_questions=[
            "Are keyboard shortcuts available?",
            "Can experienced users skip steps?",
            "Are there shortcuts for frequent actions?",
            "Can users customize their experience?"
        ],
        positive_examples=[
            "Keyboard shortcuts for power users",
            "Quick actions for common tasks",
            "Saved preferences and defaults",
            "Recent/favorite items quick access",
            "Customizable dashboard"
        ],
        negative_examples=[
            "No keyboard shortcuts",
            "Must go through wizard every time",
            "No way to save preferences",
            "One-size-fits-all experience"
        ],
        evaluation_tips=[
            "Look for accelerators for power users",
            "Check if common tasks have shortcuts",
            "Verify customization options exist"
        ]
    ),
    8: Heuristic(
        number=8,
        name="Aesthetic and Minimalist Design",
        description="Interfaces should not contain information which is irrelevant or rarely needed. Every extra unit of information competes with relevant information.",
        key_questions=[
            "Is content focused on essentials?",
            "Is visual hierarchy clear?",
            "Are there unnecessary elements?",
            "Is the design clean and uncluttered?"
        ],
        positive_examples=[
            "Progressive disclosure (show more on demand)",
            "Clean layouts with ample whitespace",
            "Clear visual hierarchy",
            "Focused content per screen",
            "Minimal but sufficient UI"
        ],
        negative_examples=[
            "Cluttered interfaces",
            "Walls of text",
            "Too many competing elements",
            "Decorative elements that don't add value",
            "Every feature visible at once"
        ],
        evaluation_tips=[
            "Question every element: is it necessary?",
            "Check for visual clutter",
            "Verify information hierarchy is clear"
        ]
    ),
    9: Heuristic(
        number=9,
        name="Help Users Recognize, Diagnose, and Recover from Errors",
        description="Error messages should be expressed in plain language, precisely indicate the problem, and constructively suggest a solution.",
        key_questions=[
            "Are error messages in plain language?",
            "Do errors explain what went wrong?",
            "Do errors suggest how to fix the problem?",
            "Are errors visible and noticeable?"
        ],
        positive_examples=[
            "'Password must be at least 8 characters' vs 'Invalid input'",
            "Error message next to problematic field",
            "Suggested corrections for common mistakes",
            "Clear recovery steps provided"
        ],
        negative_examples=[
            "'Error 404' with no explanation",
            "Generic 'Something went wrong'",
            "Error message without solution",
            "Error indicator far from problem"
        ],
        evaluation_tips=[
            "Trigger every possible error",
            "Check if error messages are helpful",
            "Verify error location and visibility"
        ]
    ),
    10: Heuristic(
        number=10,
        name="Help and Documentation",
        description="It's best if the system can be used without documentation, but it may be necessary to provide help. Documentation should be easy to search, focused on tasks, and not too large.",
        key_questions=[
            "Is help easily accessible?",
            "Is help task-focused, not feature-focused?",
            "Is help searchable?",
            "Is in-context help available?"
        ],
        positive_examples=[
            "Tooltips on complex elements",
            "Contextual help icons",
            "Searchable help center",
            "Onboarding tutorials",
            "FAQ for common questions"
        ],
        negative_examples=[
            "No help available",
            "Help buried in footer",
            "Dense documentation not searchable",
            "Help not relevant to current context"
        ],
        evaluation_tips=[
            "Look for help options throughout",
            "Check if help is easy to find",
            "Verify help content is useful"
        ]
    )
}
```

## Conducting Evaluations

### Evaluation Process

```python
@dataclass
class HeuristicFinding:
    """Single usability finding from evaluation"""

    id: str
    heuristic_violated: int  # 1-10
    location: str  # Page/screen/component
    description: str
    screenshot_url: str = ""
    severity: SeverityRating = SeverityRating.MODERATE
    recommendation: str = ""
    notes: str = ""

class HeuristicEvaluation:
    """
    Conduct heuristic evaluation

    Recommended: 3-5 evaluators independently, then combine findings
    """

    def __init__(self, product_name: str, evaluator: str):
        self.product = product_name
        self.evaluator = evaluator
        self.findings: List[HeuristicFinding] = []
        self.date = None

    def add_finding(self, finding: HeuristicFinding):
        self.findings.append(finding)

    def findings_by_heuristic(self) -> Dict[int, List[HeuristicFinding]]:
        """Group findings by heuristic violated"""
        grouped = {i: [] for i in range(1, 11)}
        for finding in self.findings:
            grouped[finding.heuristic_violated].append(finding)
        return grouped

    def findings_by_severity(self) -> Dict[SeverityRating, List[HeuristicFinding]]:
        """Group findings by severity"""
        grouped = {s: [] for s in SeverityRating}
        for finding in self.findings:
            grouped[finding.severity].append(finding)
        return grouped

    def summary_statistics(self) -> Dict:
        """Generate evaluation summary"""
        by_severity = self.findings_by_severity()
        by_heuristic = self.findings_by_heuristic()

        return {
            'total_findings': len(self.findings),
            'by_severity': {
                s.name: len(findings)
                for s, findings in by_severity.items()
            },
            'most_violated_heuristics': sorted(
                [(h, len(f)) for h, f in by_heuristic.items()],
                key=lambda x: -x[1]
            )[:3],
            'critical_issues': len(by_severity[SeverityRating.CATASTROPHIC]),
            'major_issues': len(by_severity[SeverityRating.MAJOR])
        }

    def generate_report(self) -> str:
        """Generate markdown report"""
        stats = self.summary_statistics()

        report = f"""# Heuristic Evaluation Report

## Product: {self.product}
## Evaluator: {self.evaluator}
## Date: {self.date}

---

## Executive Summary

- **Total Issues Found**: {stats['total_findings']}
- **Critical Issues**: {stats['critical_issues']}
- **Major Issues**: {stats['major_issues']}

### Most Violated Heuristics
"""
        for heuristic_num, count in stats['most_violated_heuristics']:
            heuristic = NIELSEN_HEURISTICS[heuristic_num]
            report += f"- H{heuristic_num}: {heuristic.name} ({count} issues)\n"

        report += "\n---\n\n## Detailed Findings\n\n"

        for severity in [SeverityRating.CATASTROPHIC, SeverityRating.MAJOR,
                        SeverityRating.MODERATE, SeverityRating.MINOR]:
            findings = self.findings_by_severity()[severity]
            if findings:
                report += f"### {severity.name} Issues\n\n"
                for f in findings:
                    heuristic = NIELSEN_HEURISTICS[f.heuristic_violated]
                    report += f"""#### {f.id}: {f.location}

**Heuristic Violated**: H{f.heuristic_violated} - {heuristic.name}

**Description**: {f.description}

**Recommendation**: {f.recommendation}

---

"""
        return report


# Example evaluation
evaluation = HeuristicEvaluation(
    product_name="Example App",
    evaluator="UX Specialist"
)

evaluation.add_finding(HeuristicFinding(
    id="F001",
    heuristic_violated=1,
    location="Checkout Page",
    description="No loading indicator when processing payment",
    severity=SeverityRating.MAJOR,
    recommendation="Add spinner and 'Processing payment...' message"
))
```

### Severity Rating Guidelines

```python
SEVERITY_GUIDELINES = {
    SeverityRating.COSMETIC: {
        'criteria': [
            "Only affects aesthetics",
            "Doesn't impact task completion",
            "Users unlikely to notice"
        ],
        'fix_priority': "Nice to have",
        'example': "Slightly inconsistent button padding"
    },
    SeverityRating.MINOR: {
        'criteria': [
            "Minor inconvenience",
            "Workaround exists",
            "Affects few users or rare scenarios"
        ],
        'fix_priority': "Low priority",
        'example': "Have to scroll to find secondary action"
    },
    SeverityRating.MODERATE: {
        'criteria': [
            "Significant annoyance",
            "May cause errors or confusion",
            "Affects many users"
        ],
        'fix_priority': "Should fix",
        'example': "Unclear error message - users can recover but frustrated"
    },
    SeverityRating.MAJOR: {
        'criteria': [
            "Prevents task completion for some",
            "Causes significant errors",
            "Affects critical user flows"
        ],
        'fix_priority': "Must fix soon",
        'example': "Form validation unclear - many users submit incorrectly"
    },
    SeverityRating.CATASTROPHIC: {
        'criteria': [
            "Prevents task completion entirely",
            "Causes data loss",
            "Complete failure of critical function"
        ],
        'fix_priority': "Must fix immediately",
        'example': "Submit button doesn't work - no one can complete checkout"
    }
}

def rate_severity(
    frequency: str,  # rare, occasional, frequent
    impact: str,  # low, medium, high
    persistence: str  # recoverable, difficult, impossible
) -> SeverityRating:
    """
    Calculate severity based on factors

    Frequency × Impact × Persistence = Severity
    """
    severity_matrix = {
        ('rare', 'low', 'recoverable'): SeverityRating.COSMETIC,
        ('rare', 'low', 'difficult'): SeverityRating.MINOR,
        ('rare', 'medium', 'recoverable'): SeverityRating.MINOR,
        ('occasional', 'low', 'recoverable'): SeverityRating.MINOR,
        ('occasional', 'medium', 'recoverable'): SeverityRating.MODERATE,
        ('frequent', 'low', 'recoverable'): SeverityRating.MODERATE,
        ('occasional', 'medium', 'difficult'): SeverityRating.MODERATE,
        ('frequent', 'medium', 'recoverable'): SeverityRating.MAJOR,
        ('occasional', 'high', 'recoverable'): SeverityRating.MAJOR,
        ('frequent', 'medium', 'difficult'): SeverityRating.MAJOR,
        ('frequent', 'high', 'recoverable'): SeverityRating.MAJOR,
        ('occasional', 'high', 'difficult'): SeverityRating.CATASTROPHIC,
        ('frequent', 'high', 'difficult'): SeverityRating.CATASTROPHIC,
        ('rare', 'high', 'impossible'): SeverityRating.CATASTROPHIC,
        ('occasional', 'medium', 'impossible'): SeverityRating.CATASTROPHIC,
        ('frequent', 'high', 'impossible'): SeverityRating.CATASTROPHIC,
    }

    return severity_matrix.get(
        (frequency, impact, persistence),
        SeverityRating.MODERATE
    )
```

## Cognitive Walkthrough

### Walkthrough Process

```python
@dataclass
class CognitiveWalkthroughStep:
    """Single step in cognitive walkthrough"""

    step_number: int
    task_description: str
    correct_action: str

    # Evaluation questions (answered for each step)
    q1_goal_visible: bool = False  # Will user try to achieve the goal?
    q1_notes: str = ""

    q2_action_visible: bool = False  # Will user notice the correct action?
    q2_notes: str = ""

    q3_action_associated: bool = False  # Will user associate action with goal?
    q3_notes: str = ""

    q4_feedback_clear: bool = False  # Will user understand feedback?
    q4_notes: str = ""

    issues: List[str] = None

    def __post_init__(self):
        self.issues = self.issues or []

class CognitiveWalkthrough:
    """
    Cognitive Walkthrough: Step-by-step task analysis

    Evaluates learnability for first-time/infrequent users
    Focus: "Can users figure out how to use this?"
    """

    def __init__(
        self,
        product: str,
        task: str,
        user_persona: str,
        user_goal: str
    ):
        self.product = product
        self.task = task
        self.persona = user_persona
        self.goal = user_goal
        self.steps: List[CognitiveWalkthroughStep] = []

    def add_step(self, step: CognitiveWalkthroughStep):
        self.steps.append(step)

    def evaluate_step(self, step: CognitiveWalkthroughStep) -> List[str]:
        """
        Answer four questions for each step

        Based on Wharton et al. (1994) cognitive walkthrough method
        """
        issues = []

        # Q1: Will users try to achieve the right effect?
        if not step.q1_goal_visible:
            issues.append(f"Step {step.step_number}: User may not recognize this is needed for their goal. {step.q1_notes}")

        # Q2: Will users notice the correct action is available?
        if not step.q2_action_visible:
            issues.append(f"Step {step.step_number}: Correct action not visible/obvious. {step.q2_notes}")

        # Q3: Will users associate the action with their goal?
        if not step.q3_action_associated:
            issues.append(f"Step {step.step_number}: Action-goal connection unclear. {step.q3_notes}")

        # Q4: Will users understand the feedback?
        if not step.q4_feedback_clear:
            issues.append(f"Step {step.step_number}: Feedback unclear after action. {step.q4_notes}")

        return issues

    def run_walkthrough(self) -> Dict:
        """Execute complete walkthrough"""
        all_issues = []

        for step in self.steps:
            step_issues = self.evaluate_step(step)
            step.issues = step_issues
            all_issues.extend(step_issues)

        return {
            'total_steps': len(self.steps),
            'steps_with_issues': len([s for s in self.steps if s.issues]),
            'total_issues': len(all_issues),
            'issues_by_step': {s.step_number: s.issues for s in self.steps},
            'all_issues': all_issues
        }


# Example walkthrough
checkout_walkthrough = CognitiveWalkthrough(
    product="E-commerce Site",
    task="Purchase an item",
    user_persona="First-time shopper",
    user_goal="Buy a laptop and have it delivered"
)

checkout_walkthrough.add_step(CognitiveWalkthroughStep(
    step_number=1,
    task_description="Find product",
    correct_action="Use search bar or browse categories",
    q1_goal_visible=True,
    q2_action_visible=True,
    q2_notes="Search bar is prominent in header",
    q3_action_associated=True,
    q4_feedback_clear=True
))

checkout_walkthrough.add_step(CognitiveWalkthroughStep(
    step_number=2,
    task_description="Add to cart",
    correct_action="Click 'Add to Cart' button",
    q1_goal_visible=True,
    q2_action_visible=True,
    q3_action_associated=True,
    q4_feedback_clear=False,
    q4_notes="Cart icon updates but user might not notice - no modal confirmation"
))
```

## Pluralistic Walkthrough

```python
class PluralisticWalkthrough:
    """
    Pluralistic Walkthrough: Group inspection method

    Involves: Users, developers, usability experts
    All review design together, discuss each step
    """

    def __init__(self, product: str, task: str):
        self.product = product
        self.task = task
        self.participants = []
        self.steps = []
        self.discussion_notes = {}

    def add_participant(self, name: str, role: str):
        """
        Roles: user, developer, designer, usability_expert
        """
        self.participants.append({'name': name, 'role': role})

    def add_step(self, step_number: int, screen_description: str, expected_action: str):
        self.steps.append({
            'number': step_number,
            'screen': screen_description,
            'expected_action': expected_action,
            'participant_actions': {},  # What each participant would do
            'discussion': ''
        })

    def record_participant_action(
        self,
        step_number: int,
        participant_name: str,
        action_they_would_take: str,
        confidence: int  # 1-5
    ):
        """Record what each participant would do at this step"""
        step = next(s for s in self.steps if s['number'] == step_number)
        step['participant_actions'][participant_name] = {
            'action': action_they_would_take,
            'confidence': confidence,
            'matches_expected': action_they_would_take == step['expected_action']
        }

    def analyze_step(self, step_number: int) -> Dict:
        """Analyze agreement and issues at a step"""
        step = next(s for s in self.steps if s['number'] == step_number)
        actions = step['participant_actions']

        # Calculate agreement
        action_counts = {}
        for p, data in actions.items():
            action = data['action']
            action_counts[action] = action_counts.get(action, 0) + 1

        most_common = max(action_counts.items(), key=lambda x: x[1])

        return {
            'step': step_number,
            'expected': step['expected_action'],
            'most_common_action': most_common[0],
            'agreement_rate': most_common[1] / len(actions),
            'matches_expected': most_common[0] == step['expected_action'],
            'all_actions': action_counts
        }


PLURALISTIC_PROCESS = """
## Pluralistic Walkthrough Process

1. **Preparation**
   - Create paper prototype or screenshots
   - Define user task and scenario
   - Recruit 4-6 diverse participants

2. **Session (2-3 hours)**
   - Present scenario to group
   - Show first screen, everyone writes what they'd do
   - Discuss: users first, then developers, then experts
   - Moderator records issues and insights
   - Move to next screen, repeat

3. **Analysis**
   - Compile all issues found
   - Prioritize by frequency and severity
   - Create action items

## Benefits
- Combines multiple perspectives
- Users explain their thinking
- Developers hear user confusion directly
- Quick and low-cost
"""
```

## UX Audit Framework

```python
class UXAudit:
    """
    Comprehensive UX audit framework

    Combines multiple inspection methods
    """

    def __init__(self, product: str, scope: str):
        self.product = product
        self.scope = scope  # 'full', 'feature', 'flow'
        self.audit_areas = {}

    AUDIT_FRAMEWORK = {
        'usability': {
            'methods': ['Heuristic evaluation', 'Cognitive walkthrough'],
            'criteria': [
                "Task completion efficiency",
                "Learnability",
                "Error handling",
                "Navigation clarity"
            ]
        },
        'accessibility': {
            'methods': ['WCAG checklist', 'Screen reader testing'],
            'criteria': [
                "Keyboard navigability",
                "Color contrast",
                "Screen reader compatibility",
                "Focus management"
            ]
        },
        'visual_design': {
            'methods': ['Design system review', 'Competitor analysis'],
            'criteria': [
                "Consistency",
                "Visual hierarchy",
                "Brand alignment",
                "Responsive design"
            ]
        },
        'content': {
            'methods': ['Content audit', 'Readability analysis'],
            'criteria': [
                "Clarity and conciseness",
                "Appropriate tone",
                "Helpful error messages",
                "Scannable structure"
            ]
        },
        'performance': {
            'methods': ['Core Web Vitals', 'Performance testing'],
            'criteria': [
                "Page load time",
                "Interactivity (FID/INP)",
                "Visual stability (CLS)",
                "Mobile performance"
            ]
        }
    }

    def create_audit_checklist(self) -> Dict:
        """Generate comprehensive audit checklist"""
        checklist = {}

        for area, details in self.AUDIT_FRAMEWORK.items():
            checklist[area] = {
                'methods': details['methods'],
                'items': [
                    {'criterion': c, 'status': 'not_evaluated', 'notes': '', 'severity': None}
                    for c in details['criteria']
                ]
            }

        return checklist

    def generate_audit_report(self, findings: Dict) -> str:
        """Generate audit report from findings"""
        report = f"""# UX Audit Report: {self.product}

## Scope: {self.scope}

---

## Executive Summary

[High-level findings and recommendations]

---

"""
        for area, data in findings.items():
            report += f"## {area.title()}\n\n"
            report += f"**Methods Used**: {', '.join(data['methods'])}\n\n"

            # Summary by severity
            severities = {'critical': 0, 'major': 0, 'minor': 0, 'pass': 0}
            for item in data['items']:
                if item['severity']:
                    severities[item['severity']] = severities.get(item['severity'], 0) + 1
                elif item['status'] == 'pass':
                    severities['pass'] += 1

            report += f"| Status | Count |\n|--------|-------|\n"
            for sev, count in severities.items():
                report += f"| {sev.title()} | {count} |\n"

            report += "\n### Detailed Findings\n\n"
            for item in data['items']:
                status_emoji = {'pass': '✅', 'critical': '🔴', 'major': '🟠', 'minor': '🟡'}.get(item['severity'] or item['status'], '⚪')
                report += f"- {status_emoji} **{item['criterion']}**: {item['notes']}\n"

            report += "\n---\n\n"

        return report
```

## Best Practices

1. **Use multiple evaluators**: 3-5 evaluators find 75%+ of issues
2. **Evaluate independently first**: Avoid groupthink
3. **Be specific**: Vague findings can't be fixed
4. **Provide solutions**: Don't just identify problems
5. **Prioritize ruthlessly**: Not all issues equal
6. **Combine methods**: Heuristics + walkthrough + user testing

## Common Pitfalls

- **Evaluator bias**: Personal preferences ≠ usability issues
- **Surface-level review**: Missing interaction issues
- **Too many minor issues**: Overwhelms real problems
- **No recommendations**: Findings without solutions
- **Single evaluator**: Misses 50%+ of issues
- **Skipping severity ratings**: Can't prioritize fixes

---

**Skill Type**: UX - Heuristic Evaluation
**Complexity**: Intermediate
**Typical Usage**: Design review, quality assurance, pre-launch audit
