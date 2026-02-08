---
name: ux-metrics-analytics
description: Load when user needs UX metrics, usability measurement, SUS scores, NPS, analytics implementation, or A/B testing for design decisions. Covers quantitative UX research methods.
trigger_keywords: [ux metrics, sus, system usability scale, nps, net promoter score, task success rate, time on task, error rate, csat, a/b testing, analytics, user engagement, conversion rate, retention, core web vitals, ux kpis]
---

# UX Metrics & Analytics Skill

Measuring and analyzing user experience through quantitative methods.

## Core Concepts

- **HEART Framework**: Google's Goals-Signals-Metrics approach measuring Happiness (satisfaction), Engagement (usage depth), Adoption (new users), Retention (return rate), and Task Success (completion/errors)
- **System Usability Scale (SUS)**: 10-question post-task questionnaire yielding 0-100 score; 68 is average, 80+ is excellent - enables benchmarking across products and over time
- **Task Success Rate**: Percentage of users completing defined tasks successfully; primary usability metric combining binary success with error rate and time-on-task for comprehensive view
- **Leading vs. Lagging Indicators**: Leading metrics predict future outcomes (engagement signals); lagging metrics confirm past performance (revenue, churn) - track both for actionable insights
- **Statistical Significance**: A/B tests require adequate sample size and confidence levels (typically 95%) before drawing conclusions - avoid acting on random noise

## UX Metrics Framework

### The HEART Framework

```python
from dataclasses import dataclass
from typing import List, Dict
from enum import Enum

class HEARTCategory(Enum):
    HAPPINESS = "happiness"  # Satisfaction, NPS, SUS
    ENGAGEMENT = "engagement"  # Usage frequency, depth
    ADOPTION = "adoption"  # New users, feature uptake
    RETENTION = "retention"  # Return rate, churn
    TASK_SUCCESS = "task_success"  # Completion, errors, time

@dataclass
class HEARTMetric:
    """
    Google's HEART framework for UX metrics

    Goals → Signals → Metrics
    """

    category: HEARTCategory
    goal: str  # What you want to achieve
    signal: str  # User behavior indicating progress
    metric: str  # How to measure the signal

HEART_EXAMPLES = {
    HEARTCategory.HAPPINESS: [
        HEARTMetric(
            category=HEARTCategory.HAPPINESS,
            goal="Users find the product easy to use",
            signal="Users report satisfaction in surveys",
            metric="SUS score, CSAT rating"
        ),
        HEARTMetric(
            category=HEARTCategory.HAPPINESS,
            goal="Users would recommend product",
            signal="Users express likelihood to recommend",
            metric="Net Promoter Score (NPS)"
        )
    ],
    HEARTCategory.ENGAGEMENT: [
        HEARTMetric(
            category=HEARTCategory.ENGAGEMENT,
            goal="Users actively use core features",
            signal="Frequency and depth of feature usage",
            metric="Daily/Weekly Active Users, sessions per user"
        )
    ],
    HEARTCategory.ADOPTION: [
        HEARTMetric(
            category=HEARTCategory.ADOPTION,
            goal="New users successfully onboard",
            signal="Completion of onboarding steps",
            metric="Activation rate, time to first value"
        )
    ],
    HEARTCategory.RETENTION: [
        HEARTMetric(
            category=HEARTCategory.RETENTION,
            goal="Users continue using product over time",
            signal="Users return after initial use",
            metric="Day 7/30/90 retention rate, churn rate"
        )
    ],
    HEARTCategory.TASK_SUCCESS: [
        HEARTMetric(
            category=HEARTCategory.TASK_SUCCESS,
            goal="Users complete key tasks efficiently",
            signal="Successful task completion without errors",
            metric="Task success rate, time on task, error rate"
        )
    ]
}
```

### Choosing the Right Metrics

```python
METRIC_SELECTION_GUIDE = {
    'product_stage': {
        'early_stage': {
            'focus': ['Adoption', 'Task Success'],
            'reason': "Validate product-market fit and usability"
        },
        'growth': {
            'focus': ['Engagement', 'Retention'],
            'reason': "Optimize for sustained usage"
        },
        'mature': {
            'focus': ['Happiness', 'Retention'],
            'reason': "Maintain satisfaction and reduce churn"
        }
    },
    'feature_type': {
        'new_feature': ['Adoption rate', 'Task success', 'Feature retention'],
        'redesign': ['Task success improvement', 'SUS delta', 'Error reduction'],
        'optimization': ['Time on task', 'Conversion rate', 'Engagement depth']
    }
}
```

## Usability Metrics

### System Usability Scale (SUS)

```python
class SystemUsabilityScale:
    """
    SUS: 10-question standardized usability questionnaire

    Widely used, validated, quick to administer
    Score range: 0-100 (but NOT a percentage)
    """

    QUESTIONS = [
        "I think that I would like to use this system frequently.",
        "I found the system unnecessarily complex.",
        "I thought the system was easy to use.",
        "I think that I would need support to use this system.",
        "I found the various functions well integrated.",
        "I thought there was too much inconsistency in the system.",
        "I imagine most people would learn to use this system quickly.",
        "I found the system very cumbersome to use.",
        "I felt very confident using the system.",
        "I needed to learn a lot before I could get going."
    ]

    # Questions alternate: odd = positive, even = negative

    @staticmethod
    def calculate_score(responses: List[int]) -> float:
        """
        Calculate SUS score from responses

        Responses: 1-5 Likert scale for each question

        Scoring:
        - Odd questions (positive): subtract 1
        - Even questions (negative): subtract from 5
        - Sum all, multiply by 2.5
        """
        if len(responses) != 10:
            raise ValueError("SUS requires exactly 10 responses")

        adjusted = []
        for i, response in enumerate(responses):
            if i % 2 == 0:  # Odd questions (0-indexed, so even index)
                adjusted.append(response - 1)
            else:  # Even questions (negative)
                adjusted.append(5 - response)

        return sum(adjusted) * 2.5

    @staticmethod
    def interpret_score(score: float) -> Dict:
        """
        Interpret SUS score

        Benchmarks (Bangor et al., 2009):
        - 68: Average
        - 80.3: A grade (top 10%)
        - 51: F grade (bottom 15%)
        """
        if score >= 80.3:
            grade = 'A'
            adjective = 'Excellent'
        elif score >= 68:
            grade = 'B'
            adjective = 'Good'
        elif score >= 51:
            grade = 'C'
            adjective = 'OK'
        else:
            grade = 'F'
            adjective = 'Poor'

        return {
            'score': round(score, 1),
            'grade': grade,
            'adjective': adjective,
            'percentile': 'Above average' if score >= 68 else 'Below average',
            'benchmark_comparison': {
                'vs_average': round(score - 68, 1),
                'vs_excellent': round(score - 80.3, 1)
            }
        }

# Example usage
responses = [4, 2, 5, 1, 4, 2, 5, 1, 5, 2]  # Sample responses
sus = SystemUsabilityScale()
score = sus.calculate_score(responses)
interpretation = sus.interpret_score(score)
# {'score': 85.0, 'grade': 'A', 'adjective': 'Excellent', ...}
```

### Task Metrics

```python
from datetime import datetime, timedelta
from typing import Optional

@dataclass
class TaskAttempt:
    """Record of single task attempt"""
    task_id: str
    participant_id: str
    started_at: datetime
    completed_at: Optional[datetime]
    successful: bool
    error_count: int
    assistance_requested: bool
    satisfaction_rating: Optional[int]  # 1-7 SEQ scale

class TaskMetrics:
    """Calculate task-level usability metrics"""

    def __init__(self):
        self.attempts: List[TaskAttempt] = []

    def add_attempt(self, attempt: TaskAttempt):
        self.attempts.append(attempt)

    def task_success_rate(self, task_id: str) -> float:
        """
        Percentage of users who completed task successfully

        Binary success (yes/no)
        """
        task_attempts = [a for a in self.attempts if a.task_id == task_id]
        if not task_attempts:
            return 0.0

        successful = sum(1 for a in task_attempts if a.successful)
        return successful / len(task_attempts)

    def time_on_task(self, task_id: str) -> Dict:
        """
        Time to complete task (successful attempts only)

        Report: mean, median, range
        """
        completed = [
            a for a in self.attempts
            if a.task_id == task_id and a.successful and a.completed_at
        ]

        if not completed:
            return {'mean': None, 'median': None}

        times = [
            (a.completed_at - a.started_at).total_seconds()
            for a in completed
        ]

        times.sort()
        n = len(times)

        return {
            'mean': sum(times) / n,
            'median': times[n // 2] if n % 2 else (times[n//2 - 1] + times[n//2]) / 2,
            'min': min(times),
            'max': max(times),
            'std_dev': self._std_dev(times)
        }

    def error_rate(self, task_id: str) -> float:
        """
        Average errors per task attempt

        Lower is better
        """
        task_attempts = [a for a in self.attempts if a.task_id == task_id]
        if not task_attempts:
            return 0.0

        total_errors = sum(a.error_count for a in task_attempts)
        return total_errors / len(task_attempts)

    def single_ease_question(self, task_id: str) -> float:
        """
        SEQ: Post-task satisfaction (1-7 scale)

        "How easy was this task?"
        Industry benchmark: ~5.5
        """
        rated = [
            a for a in self.attempts
            if a.task_id == task_id and a.satisfaction_rating
        ]

        if not rated:
            return 0.0

        return sum(a.satisfaction_rating for a in rated) / len(rated)

    def task_level_satisfaction(self, task_id: str) -> float:
        """
        Alias for SEQ - commonly requested metric
        """
        return self.single_ease_question(task_id)

    def _std_dev(self, values: List[float]) -> float:
        """Calculate standard deviation"""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance ** 0.5
```

## Satisfaction Metrics

### Net Promoter Score (NPS)

```python
class NetPromoterScore:
    """
    NPS: Likelihood to recommend (0-10 scale)

    "How likely are you to recommend [product] to a friend or colleague?"

    Categories:
    - Promoters (9-10): Loyal enthusiasts
    - Passives (7-8): Satisfied but unenthusiastic
    - Detractors (0-6): Unhappy, can damage brand
    """

    @staticmethod
    def categorize(score: int) -> str:
        if score >= 9:
            return 'Promoter'
        elif score >= 7:
            return 'Passive'
        else:
            return 'Detractor'

    @staticmethod
    def calculate_nps(responses: List[int]) -> Dict:
        """
        Calculate NPS from responses

        NPS = % Promoters - % Detractors
        Range: -100 to +100
        """
        if not responses:
            return {'nps': 0, 'error': 'No responses'}

        total = len(responses)
        promoters = sum(1 for r in responses if r >= 9)
        detractors = sum(1 for r in responses if r <= 6)
        passives = total - promoters - detractors

        nps = ((promoters - detractors) / total) * 100

        return {
            'nps': round(nps, 1),
            'promoters': promoters,
            'promoters_pct': round(promoters / total * 100, 1),
            'passives': passives,
            'passives_pct': round(passives / total * 100, 1),
            'detractors': detractors,
            'detractors_pct': round(detractors / total * 100, 1),
            'total_responses': total
        }

    @staticmethod
    def interpret_nps(nps: float) -> str:
        """
        Interpret NPS score

        Benchmarks vary by industry, but generally:
        """
        if nps >= 70:
            return "World-class"
        elif nps >= 50:
            return "Excellent"
        elif nps >= 30:
            return "Good"
        elif nps >= 0:
            return "Needs improvement"
        else:
            return "Critical - action required"


# Example
responses = [10, 9, 8, 7, 9, 6, 10, 8, 5, 9]
result = NetPromoterScore.calculate_nps(responses)
# {'nps': 20.0, 'promoters': 5, 'promoters_pct': 50.0, ...}
```

### Customer Satisfaction (CSAT)

```python
class CustomerSatisfaction:
    """
    CSAT: Direct satisfaction measurement

    Typically: "How satisfied are you with [X]?"
    Scale: 1-5 or 1-7
    """

    @staticmethod
    def calculate_csat(
        responses: List[int],
        scale_max: int = 5,
        satisfied_threshold: int = 4
    ) -> Dict:
        """
        Calculate CSAT percentage

        CSAT % = (Satisfied responses / Total responses) × 100
        """
        if not responses:
            return {'csat': 0, 'error': 'No responses'}

        satisfied = sum(1 for r in responses if r >= satisfied_threshold)
        total = len(responses)

        return {
            'csat_percentage': round(satisfied / total * 100, 1),
            'average_score': round(sum(responses) / total, 2),
            'satisfied_count': satisfied,
            'total_responses': total,
            'scale': f"1-{scale_max}",
            'threshold': f">= {satisfied_threshold}"
        }

    @staticmethod
    def csat_by_segment(
        responses: List[Dict],  # [{score: int, segment: str}]
        segment_field: str = 'segment'
    ) -> Dict[str, float]:
        """Calculate CSAT for each user segment"""
        from collections import defaultdict

        segments = defaultdict(list)
        for r in responses:
            segments[r[segment_field]].append(r['score'])

        return {
            segment: CustomerSatisfaction.calculate_csat(scores)['csat_percentage']
            for segment, scores in segments.items()
        }
```

### Customer Effort Score (CES)

```python
class CustomerEffortScore:
    """
    CES: Measures ease of interaction

    "How easy was it to [complete task]?"
    Scale: 1-7 (1=Very Difficult, 7=Very Easy)

    Lower effort correlates with higher loyalty
    """

    @staticmethod
    def calculate_ces(responses: List[int]) -> Dict:
        """
        Calculate CES

        Report: Average score, % low effort (6-7), % high effort (1-3)
        """
        if not responses:
            return {'ces': 0, 'error': 'No responses'}

        total = len(responses)
        low_effort = sum(1 for r in responses if r >= 6)
        high_effort = sum(1 for r in responses if r <= 3)

        return {
            'ces_average': round(sum(responses) / total, 2),
            'low_effort_pct': round(low_effort / total * 100, 1),
            'high_effort_pct': round(high_effort / total * 100, 1),
            'total_responses': total
        }
```

## Behavioral Metrics

### Engagement Metrics

```python
@dataclass
class EngagementMetrics:
    """Track user engagement patterns"""

    def calculate_dau_mau(
        self,
        daily_users: List[int],  # Daily active user counts
        monthly_users: int
    ) -> float:
        """
        DAU/MAU ratio: Stickiness metric

        Higher ratio = more engaged users
        Good: 20%+ for most apps
        Great: 50%+ (daily habit apps)
        """
        avg_dau = sum(daily_users) / len(daily_users)
        return round(avg_dau / monthly_users * 100, 1) if monthly_users > 0 else 0

    def sessions_per_user(
        self,
        total_sessions: int,
        unique_users: int,
        period: str = 'week'
    ) -> float:
        """Average sessions per user in period"""
        return round(total_sessions / unique_users, 2) if unique_users > 0 else 0

    def session_duration(
        self,
        durations: List[float]  # In seconds
    ) -> Dict:
        """Session duration statistics"""
        if not durations:
            return {}

        durations.sort()
        n = len(durations)

        return {
            'mean': round(sum(durations) / n, 1),
            'median': durations[n // 2],
            'p90': durations[int(n * 0.9)],
            'p95': durations[int(n * 0.95)]
        }

    def feature_adoption(
        self,
        users_using_feature: int,
        total_active_users: int
    ) -> float:
        """Percentage of users who have used a feature"""
        return round(users_using_feature / total_active_users * 100, 1)

    def depth_of_engagement(
        self,
        feature_usage_counts: Dict[str, int],
        total_users: int
    ) -> Dict:
        """How many features each user engages with"""
        # Users engaging with 1, 2, 3+ features
        return {
            feature: round(count / total_users * 100, 1)
            for feature, count in feature_usage_counts.items()
        }
```

### Retention Metrics

```python
class RetentionMetrics:
    """Track user retention patterns"""

    @staticmethod
    def cohort_retention(
        cohort_users: List[str],  # Users who signed up in cohort period
        active_users_by_period: Dict[int, List[str]]  # {period_number: [active_user_ids]}
    ) -> Dict[int, float]:
        """
        Calculate retention by cohort

        Returns: {period: retention_rate}
        """
        cohort_size = len(cohort_users)
        cohort_set = set(cohort_users)

        retention = {}
        for period, active in active_users_by_period.items():
            retained = len(set(active) & cohort_set)
            retention[period] = round(retained / cohort_size * 100, 1)

        return retention

    @staticmethod
    def rolling_retention(
        user_first_seen: Dict[str, datetime],
        user_last_seen: Dict[str, datetime],
        days: int = 30
    ) -> float:
        """
        Rolling retention: % of users active in last N days

        vs. classic retention which is point-in-time
        """
        cutoff = datetime.now() - timedelta(days=days)
        total_users = len(user_first_seen)
        active_users = sum(
            1 for user_id, last_seen in user_last_seen.items()
            if last_seen >= cutoff
        )

        return round(active_users / total_users * 100, 1)

    @staticmethod
    def churn_rate(
        users_start: int,
        users_end: int,
        new_users: int,
        period: str = 'month'
    ) -> float:
        """
        Churn rate: % of users lost in period

        Churn = (Users_start + New - Users_end) / Users_start
        """
        churned = users_start + new_users - users_end
        return round(churned / users_start * 100, 1) if users_start > 0 else 0
```

## A/B Testing for UX

### Test Design

```python
from scipy import stats
import numpy as np

class ABTestDesign:
    """Design and analyze A/B tests for UX changes"""

    @staticmethod
    def calculate_sample_size(
        baseline_rate: float,  # Current conversion/success rate
        minimum_detectable_effect: float,  # Relative change to detect (e.g., 0.1 = 10%)
        alpha: float = 0.05,  # Significance level
        power: float = 0.8  # Statistical power
    ) -> int:
        """
        Calculate required sample size per variant

        For binary outcomes (conversion, task success)
        """
        from statsmodels.stats.power import NormalIndPower

        effect_size = baseline_rate * minimum_detectable_effect / np.sqrt(
            baseline_rate * (1 - baseline_rate)
        )

        analysis = NormalIndPower()
        sample_size = analysis.solve_power(
            effect_size=effect_size,
            alpha=alpha,
            power=power,
            alternative='two-sided'
        )

        return int(np.ceil(sample_size))

    @staticmethod
    def analyze_binary_outcome(
        control_successes: int,
        control_total: int,
        treatment_successes: int,
        treatment_total: int
    ) -> Dict:
        """
        Analyze A/B test with binary outcome (conversion, task success)

        Uses chi-square test
        """
        # Conversion rates
        control_rate = control_successes / control_total
        treatment_rate = treatment_successes / treatment_total
        relative_lift = (treatment_rate - control_rate) / control_rate * 100

        # Chi-square test
        contingency = [
            [control_successes, control_total - control_successes],
            [treatment_successes, treatment_total - treatment_successes]
        ]
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency)

        return {
            'control_rate': round(control_rate * 100, 2),
            'treatment_rate': round(treatment_rate * 100, 2),
            'relative_lift': round(relative_lift, 2),
            'absolute_lift': round((treatment_rate - control_rate) * 100, 2),
            'p_value': round(p_value, 4),
            'statistically_significant': p_value < 0.05,
            'confidence': round((1 - p_value) * 100, 1)
        }

    @staticmethod
    def analyze_continuous_outcome(
        control_values: List[float],
        treatment_values: List[float]
    ) -> Dict:
        """
        Analyze A/B test with continuous outcome (time on task, SUS score)

        Uses t-test
        """
        control_mean = np.mean(control_values)
        treatment_mean = np.mean(treatment_values)

        # Independent samples t-test
        t_stat, p_value = stats.ttest_ind(control_values, treatment_values)

        return {
            'control_mean': round(control_mean, 2),
            'treatment_mean': round(treatment_mean, 2),
            'difference': round(treatment_mean - control_mean, 2),
            'percent_change': round((treatment_mean - control_mean) / control_mean * 100, 2),
            't_statistic': round(t_stat, 3),
            'p_value': round(p_value, 4),
            'statistically_significant': p_value < 0.05
        }
```

### UX-Specific A/B Considerations

```python
UX_AB_TEST_GUIDELINES = {
    'what_to_test': [
        "Navigation structure changes",
        "Form redesigns",
        "CTA button variations",
        "Onboarding flow alternatives",
        "Information architecture changes"
    ],
    'metrics_to_track': {
        'primary': "Task success rate, conversion rate",
        'secondary': "Time on task, error rate, satisfaction",
        'guardrails': "Engagement, retention (don't regress)"
    },
    'common_mistakes': [
        "Running test too short (novelty effect)",
        "Not accounting for learning effects",
        "Ignoring qualitative feedback",
        "Testing too many changes at once",
        "Not segmenting by user type"
    ],
    'minimum_duration': "2 full business cycles (usually 2-4 weeks)",
    'sample_considerations': [
        "New vs returning users may respond differently",
        "Mobile vs desktop may need separate analysis",
        "Consider user segments separately"
    ]
}
```

## Analytics Implementation

### Event Tracking Schema

```python
@dataclass
class UXEvent:
    """Structured UX event for analytics"""

    # Required
    event_name: str
    timestamp: datetime
    user_id: str
    session_id: str

    # Context
    page_url: str
    page_title: str
    component: Optional[str] = None

    # UX-specific
    interaction_type: str = ""  # click, view, scroll, form_submit
    element_id: Optional[str] = None
    element_text: Optional[str] = None

    # Task tracking
    task_id: Optional[str] = None
    task_step: Optional[int] = None
    task_success: Optional[bool] = None

    # Custom properties
    properties: Dict = field(default_factory=dict)


UX_EVENT_TAXONOMY = {
    'navigation': [
        'menu_opened',
        'menu_closed',
        'nav_item_clicked',
        'breadcrumb_clicked',
        'search_initiated'
    ],
    'engagement': [
        'page_viewed',
        'scroll_depth_reached',  # 25%, 50%, 75%, 100%
        'time_on_page_milestone',  # 30s, 60s, 120s
        'element_visible',  # Impression tracking
        'content_expanded'
    ],
    'conversion': [
        'cta_clicked',
        'form_started',
        'form_field_completed',
        'form_submitted',
        'form_abandoned',
        'checkout_started',
        'purchase_completed'
    ],
    'task_flow': [
        'task_started',
        'task_step_completed',
        'task_error',
        'task_abandoned',
        'task_completed'
    ],
    'feedback': [
        'rating_submitted',
        'survey_started',
        'survey_completed',
        'feedback_provided'
    ]
}
```

### Analytics Dashboard Metrics

```python
UX_DASHBOARD_METRICS = {
    'acquisition': {
        'metrics': ['New users', 'Traffic sources', 'Landing page performance'],
        'questions': "Where do users come from? What's their first impression?"
    },
    'activation': {
        'metrics': ['Signup completion rate', 'Time to first action', 'Activation rate'],
        'questions': "Do users complete onboarding? How quickly do they get value?"
    },
    'engagement': {
        'metrics': ['DAU/MAU', 'Sessions per user', 'Feature adoption', 'Session duration'],
        'questions': "How often do users return? What features do they use?"
    },
    'retention': {
        'metrics': ['Day 1/7/30 retention', 'Cohort retention curves', 'Churn rate'],
        'questions': "Do users stick around? When do they drop off?"
    },
    'task_success': {
        'metrics': ['Task completion rate', 'Error rate', 'Time on task'],
        'questions': "Can users accomplish their goals? Where do they struggle?"
    },
    'satisfaction': {
        'metrics': ['NPS', 'CSAT', 'SUS', 'CES'],
        'questions': "Are users satisfied? Would they recommend us?"
    }
}
```

## Reporting & Communication

### Metric Reporting Template

```python
METRIC_REPORT_TEMPLATE = """
# UX Metrics Report: {period}

## Executive Summary
- **Key Highlight**: {highlight}
- **Trend**: {trend_direction} from previous period
- **Action Needed**: {action_items}

## Core Metrics

### Satisfaction
| Metric | Current | Previous | Change | Target |
|--------|---------|----------|--------|--------|
| NPS | {nps} | {nps_prev} | {nps_change} | {nps_target} |
| SUS | {sus} | {sus_prev} | {sus_change} | {sus_target} |
| CSAT | {csat}% | {csat_prev}% | {csat_change} | {csat_target}% |

### Task Success
| Task | Success Rate | Time (median) | Error Rate |
|------|--------------|---------------|------------|
| {task_1} | {rate_1}% | {time_1}s | {errors_1} |
| {task_2} | {rate_2}% | {time_2}s | {errors_2} |

### Engagement
- DAU/MAU Ratio: {dau_mau}%
- Sessions/User (weekly): {sessions_per_user}
- Feature Adoption:
  - {feature_1}: {adoption_1}%
  - {feature_2}: {adoption_2}%

## Insights & Recommendations

### What's Working
- {positive_insight_1}
- {positive_insight_2}

### Areas for Improvement
- {improvement_area_1}: {recommendation_1}
- {improvement_area_2}: {recommendation_2}

### Next Steps
1. {action_1}
2. {action_2}
3. {action_3}
"""
```

## Best Practices

1. **Balance qual and quant**: Numbers need context from user feedback
2. **Track trends, not snapshots**: Single measurements are noisy
3. **Segment your data**: Overall metrics hide important differences
4. **Set baselines first**: Can't improve what you don't measure
5. **Focus on actionable metrics**: Vanity metrics don't drive decisions
6. **Validate with qualitative**: Numbers tell what, research tells why

## Common Pitfalls

- **Over-tracking**: Too many metrics = analysis paralysis
- **Ignoring sample size**: Small samples = unreliable conclusions
- **Survivorship bias**: Only measuring users who didn't churn
- **Conflating correlation and causation**: A/B tests prove causation
- **Metric gaming**: Optimizing for metric instead of user experience
- **Delayed measurement**: Waiting too long to implement analytics

---

**Skill Type**: UX - Metrics & Analytics
**Complexity**: Intermediate to Advanced
**Typical Usage**: Product decisions, A/B testing, UX research
