---
name: user-research-methods
description: Load when user needs user research techniques including interviews, surveys, usability testing, contextual inquiry, diary studies, or card sorting. Covers qualitative and quantitative research methods.
trigger_keywords: [user research, user interview, usability testing, contextual inquiry, diary study, card sorting, tree testing, survey design, research plan, recruitment, moderated testing, unmoderated testing, think aloud, affinity mapping]
---

# User Research Methods Skill

Comprehensive user research techniques for understanding user needs, behaviors, and pain points.

## Core Concepts

- **Generative vs. Evaluative Research**: Generative research discovers user needs and opportunities (interviews, contextual inquiry); evaluative research tests solutions (usability testing, A/B tests)
- **Qualitative vs. Quantitative**: Qualitative answers "why" with depth (5-8 participants often sufficient); quantitative answers "how many" with statistical significance (larger samples needed)
- **Research Triangulation**: Combine multiple methods (interviews + analytics + surveys) to validate findings - single-method research risks missing the full picture
- **Participant Recruitment**: Screener criteria must match target users; incentives should be appropriate; 5 users find 85% of usability issues in focused testing
- **Research Ethics**: Informed consent, data privacy, right to withdraw, avoiding leading questions - ethical research protects participants and produces valid insights

## Research Planning

### Research Plan Template

```markdown
# Research Plan: [Project Name]

## Background
- Product/feature context
- What we know already
- Stakeholder questions

## Research Questions
1. Primary: What are we trying to learn?
2. Secondary: Supporting questions

## Methodology
- Method(s) selected and rationale
- Sample size and criteria
- Timeline and milestones

## Recruitment
- Target participants
- Screening criteria
- Incentive structure

## Deliverables
- Format of findings
- Stakeholder presentation
- Timeline for synthesis
```

### Method Selection Framework

```python
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict

class ResearchGoal(Enum):
    DISCOVER = "discover"      # Understand problem space
    EXPLORE = "explore"        # Generate ideas/concepts
    EVALUATE = "evaluate"      # Test existing designs
    LISTEN = "listen"          # Ongoing feedback

class DataType(Enum):
    QUALITATIVE = "qualitative"  # Rich, contextual
    QUANTITATIVE = "quantitative"  # Measurable, statistical

@dataclass
class ResearchMethod:
    name: str
    goal: ResearchGoal
    data_type: DataType
    sample_size: str  # Typical range
    time_investment: str
    cost: str  # Low/Medium/High
    when_to_use: List[str]
    limitations: List[str]

METHODS = {
    'user_interview': ResearchMethod(
        name="User Interview",
        goal=ResearchGoal.DISCOVER,
        data_type=DataType.QUALITATIVE,
        sample_size="5-12 participants",
        time_investment="1-2 weeks",
        cost="Medium",
        when_to_use=[
            "Understanding motivations and mental models",
            "Exploring problem space",
            "Learning about current workflows",
            "Getting rich contextual data"
        ],
        limitations=[
            "Self-reported data may differ from behavior",
            "Time-intensive",
            "Interviewer bias possible"
        ]
    ),
    'usability_test': ResearchMethod(
        name="Usability Testing",
        goal=ResearchGoal.EVALUATE,
        data_type=DataType.QUALITATIVE,
        sample_size="5-8 per round",
        time_investment="1-2 weeks",
        cost="Medium",
        when_to_use=[
            "Testing specific flows/tasks",
            "Finding usability issues",
            "Comparing design alternatives",
            "Validating before launch"
        ],
        limitations=[
            "Lab setting may affect behavior",
            "Limited to testable prototypes",
            "Task-focused, may miss broader issues"
        ]
    ),
    'survey': ResearchMethod(
        name="Survey",
        goal=ResearchGoal.LISTEN,
        data_type=DataType.QUANTITATIVE,
        sample_size="100-1000+ responses",
        time_investment="1-3 weeks",
        cost="Low",
        when_to_use=[
            "Quantifying behaviors/attitudes",
            "Reaching large sample",
            "Validating qualitative findings",
            "Tracking trends over time"
        ],
        limitations=[
            "No follow-up questions",
            "Survey fatigue",
            "Sampling bias"
        ]
    ),
    'contextual_inquiry': ResearchMethod(
        name="Contextual Inquiry",
        goal=ResearchGoal.DISCOVER,
        data_type=DataType.QUALITATIVE,
        sample_size="4-8 participants",
        time_investment="2-4 weeks",
        cost="High",
        when_to_use=[
            "Understanding real environment",
            "Complex workflows",
            "Enterprise/B2B products",
            "Physical + digital interactions"
        ],
        limitations=[
            "Very time-intensive",
            "Travel may be required",
            "Presence may affect behavior"
        ]
    ),
    'diary_study': ResearchMethod(
        name="Diary Study",
        goal=ResearchGoal.DISCOVER,
        data_type=DataType.QUALITATIVE,
        sample_size="10-20 participants",
        time_investment="2-6 weeks",
        cost="Medium",
        when_to_use=[
            "Longitudinal behavior patterns",
            "Infrequent events",
            "Emotional experiences",
            "Real-world context"
        ],
        limitations=[
            "Participant fatigue/dropout",
            "Relies on self-reporting",
            "Analysis is time-intensive"
        ]
    ),
    'card_sorting': ResearchMethod(
        name="Card Sorting",
        goal=ResearchGoal.EXPLORE,
        data_type=DataType.QUALITATIVE,
        sample_size="15-30 participants",
        time_investment="1-2 weeks",
        cost="Low",
        when_to_use=[
            "Information architecture",
            "Navigation structure",
            "Category naming",
            "Mental model alignment"
        ],
        limitations=[
            "Artificial task",
            "Doesn't test findability",
            "May not match real behavior"
        ]
    )
}

def recommend_method(
    goal: ResearchGoal,
    budget: str,
    timeline_weeks: int,
    need_quantitative: bool = False
) -> List[str]:
    """Recommend research methods based on constraints"""
    recommendations = []

    cost_map = {'Low': 1, 'Medium': 2, 'High': 3}
    budget_limit = cost_map.get(budget, 2)

    for name, method in METHODS.items():
        if method.goal == goal or goal == ResearchGoal.DISCOVER:
            if cost_map[method.cost] <= budget_limit:
                if need_quantitative and method.data_type == DataType.QUANTITATIVE:
                    recommendations.insert(0, name)
                elif not need_quantitative:
                    recommendations.append(name)

    return recommendations[:3]
```

## User Interviews

### Interview Guide Structure

```python
class InterviewGuide:
    """Structure for effective user interviews"""

    def __init__(self, topic: str, duration_minutes: int = 60):
        self.topic = topic
        self.duration = duration_minutes
        self.sections = []

    def build_guide(self) -> Dict:
        """
        Standard interview structure

        60-minute interview allocation:
        - Intro/rapport: 5 min
        - Background: 10 min
        - Core questions: 35 min
        - Wrap-up: 10 min
        """
        return {
            'intro': {
                'duration': 5,
                'goals': ['Build rapport', 'Set expectations', 'Get consent'],
                'script': """
                    Thank you for taking the time to speak with me today.
                    I'm [name] from [company], and we're researching [topic].

                    This conversation will take about [duration] minutes.
                    There are no right or wrong answers - we want your honest opinions.

                    Everything you share is confidential and used for research only.
                    May I record this session for note-taking purposes?

                    Do you have any questions before we begin?
                """
            },
            'background': {
                'duration': 10,
                'goals': ['Understand context', 'Build on rapport'],
                'questions': [
                    "Tell me a bit about your role and what you do day-to-day.",
                    "How long have you been doing [relevant activity]?",
                    "Walk me through a typical [day/workflow/task]."
                ]
            },
            'core': {
                'duration': 35,
                'structure': 'funnel',  # Broad → Specific
                'question_types': {
                    'experience': "Tell me about the last time you...",
                    'behavior': "Walk me through how you typically...",
                    'opinion': "What do you think about...",
                    'feeling': "How did that make you feel?",
                    'hypothetical': "If you could change one thing about..."
                }
            },
            'wrap_up': {
                'duration': 10,
                'questions': [
                    "Is there anything else you'd like to share that we haven't covered?",
                    "If you could give advice to someone designing [product], what would it be?",
                    "Who else should we talk to about this?"
                ]
            }
        }


# Question Writing Best Practices
INTERVIEW_TIPS = {
    'do': [
        "Ask open-ended questions (how, what, why, tell me about)",
        "Use their language, not jargon",
        "Follow up with 'Why?' and 'Can you tell me more?'",
        "Ask for specific examples and stories",
        "Embrace silence - let them think",
        "Mirror their words back to clarify"
    ],
    'avoid': [
        "Leading questions ('Don't you think X is better?')",
        "Double-barreled questions (two questions in one)",
        "Yes/no questions (unless confirming)",
        "Hypotheticals ('Would you use X?') - ask about past behavior",
        "Asking about future behavior predictions",
        "Interrupting or filling silences too quickly"
    ],
    'probes': [
        "Tell me more about that.",
        "Why do you say that?",
        "Can you give me an example?",
        "What happened next?",
        "How did that make you feel?",
        "What were you thinking at that moment?"
    ]
}
```

### Interview Analysis

```python
import pandas as pd
from collections import defaultdict
from typing import List, Tuple

class InterviewAnalysis:
    """Analyze interview transcripts"""

    def __init__(self):
        self.transcripts = []
        self.codes = defaultdict(list)

    def add_transcript(self, participant_id: str, text: str):
        """Add interview transcript"""
        self.transcripts.append({
            'participant': participant_id,
            'text': text
        })

    def code_passage(self, participant_id: str, passage: str, code: str):
        """
        Apply thematic code to passage

        Coding = tagging quotes with themes
        """
        self.codes[code].append({
            'participant': participant_id,
            'quote': passage
        })

    def affinity_mapping(self) -> Dict[str, List[str]]:
        """
        Group codes into higher-level themes

        Affinity mapping process:
        1. Write each insight/quote on sticky note
        2. Group similar items together
        3. Name each group (theme)
        4. Arrange themes hierarchically
        """
        themes = {}
        for code, quotes in self.codes.items():
            # Group by similarity (manual process in practice)
            theme = self._suggest_theme(code)
            if theme not in themes:
                themes[theme] = []
            themes[theme].extend([q['quote'] for q in quotes])
        return themes

    def _suggest_theme(self, code: str) -> str:
        """Suggest parent theme for code"""
        # In practice, this is manual clustering
        theme_mapping = {
            'frustration': 'Pain Points',
            'confusion': 'Pain Points',
            'workaround': 'Pain Points',
            'delight': 'Positive Experiences',
            'satisfaction': 'Positive Experiences',
            'wish': 'Unmet Needs',
            'feature_request': 'Unmet Needs'
        }
        return theme_mapping.get(code.lower(), 'General')

    def quote_frequency(self) -> pd.DataFrame:
        """How many participants mentioned each theme?"""
        freq = []
        for code, quotes in self.codes.items():
            participants = set(q['participant'] for q in quotes)
            freq.append({
                'code': code,
                'mention_count': len(quotes),
                'participant_count': len(participants),
                'sample_quote': quotes[0]['quote'] if quotes else ''
            })
        return pd.DataFrame(freq).sort_values('participant_count', ascending=False)
```

## Usability Testing

### Test Plan

```python
@dataclass
class UsabilityTestPlan:
    """Usability test planning and execution"""

    product: str
    prototype_url: str
    participant_count: int = 5
    session_duration: int = 60
    moderated: bool = True

    def create_tasks(self) -> List[Dict]:
        """
        Task design principles:
        - Realistic scenarios, not feature tours
        - Goal-oriented (what user wants to achieve)
        - No hints about UI location
        - Measurable completion criteria
        """
        return [
            {
                'task_id': 1,
                'scenario': "You want to [realistic goal]. Please try to do that now.",
                'success_criteria': ['Completed checkout', 'Confirmation shown'],
                'max_time_seconds': 180,
                'metrics': ['completion', 'time', 'errors', 'satisfaction']
            }
        ]

    def session_script(self) -> str:
        return """
        INTRODUCTION (5 min)
        ---
        Thank you for participating today. We're testing [product], not you.
        There are no wrong answers - if something is confusing, that's valuable feedback.

        I'll ask you to complete some tasks. Please think aloud as you go -
        tell me what you're looking at, what you're thinking, what you expect.

        You can ask questions, but I may not answer during tasks to see
        what you'd do on your own. I'll answer everything at the end.

        May I record the session? [Get consent]

        TASKS (45 min)
        ---
        [For each task:]
        - Read scenario
        - Start timer
        - Observe, take notes, don't help
        - Record: completion, time, errors, verbatims
        - Post-task: "How difficult was that?" (1-7 scale)

        DEBRIEF (10 min)
        ---
        - Overall impressions?
        - Most confusing part?
        - What would you change?
        - SUS questionnaire (optional)
        """


class UsabilityMetrics:
    """Calculate usability metrics from test sessions"""

    def __init__(self):
        self.sessions = []

    def add_session(self, participant_id: str, tasks: List[Dict]):
        """
        tasks: List of {task_id, completed, time_seconds, error_count, satisfaction}
        """
        self.sessions.append({
            'participant': participant_id,
            'tasks': tasks
        })

    def task_success_rate(self, task_id: int) -> float:
        """Percentage of participants who completed task"""
        completed = sum(
            1 for s in self.sessions
            for t in s['tasks']
            if t['task_id'] == task_id and t['completed']
        )
        total = len(self.sessions)
        return completed / total if total > 0 else 0

    def average_time_on_task(self, task_id: int) -> float:
        """Average completion time (successful attempts only)"""
        times = [
            t['time_seconds']
            for s in self.sessions
            for t in s['tasks']
            if t['task_id'] == task_id and t['completed']
        ]
        return sum(times) / len(times) if times else 0

    def error_rate(self, task_id: int) -> float:
        """Average errors per task attempt"""
        errors = [
            t['error_count']
            for s in self.sessions
            for t in s['tasks']
            if t['task_id'] == task_id
        ]
        return sum(errors) / len(errors) if errors else 0

    def single_ease_question(self, task_id: int) -> float:
        """
        SEQ: "How difficult was this task?" (1-7 scale)
        Industry benchmark: ~5.5
        """
        scores = [
            t['satisfaction']
            for s in self.sessions
            for t in s['tasks']
            if t['task_id'] == task_id
        ]
        return sum(scores) / len(scores) if scores else 0

    def identify_issues(self) -> List[Dict]:
        """
        Issue severity rating:
        - Critical (4): Prevents task completion
        - Serious (3): Major delay/frustration
        - Minor (2): Small annoyance
        - Cosmetic (1): Polish item
        """
        issues = []
        for session in self.sessions:
            for task in session['tasks']:
                if not task['completed'] or task['error_count'] > 0:
                    issues.append({
                        'task_id': task['task_id'],
                        'participant': session['participant'],
                        'completed': task['completed'],
                        'errors': task['error_count'],
                        'notes': task.get('notes', '')
                    })
        return issues
```

### Moderated vs Unmoderated

```python
TESTING_MODES = {
    'moderated_in_person': {
        'pros': [
            "Real-time follow-up questions",
            "Observe body language",
            "Build rapport",
            "Can clarify confusion"
        ],
        'cons': [
            "Time-intensive (1:1)",
            "Scheduling challenges",
            "Facilitator bias possible",
            "Geographic limitations"
        ],
        'best_for': [
            "Complex or sensitive topics",
            "Early-stage exploration",
            "Enterprise/B2B products",
            "Accessibility testing"
        ]
    },
    'moderated_remote': {
        'pros': [
            "No travel required",
            "Still have real-time interaction",
            "Can screen share",
            "Easier scheduling"
        ],
        'cons': [
            "Tech issues possible",
            "Harder to build rapport",
            "Miss body language",
            "Requires stable internet"
        ],
        'tools': ['Zoom', 'UserTesting', 'Lookback', 'dscout']
    },
    'unmoderated': {
        'pros': [
            "Scale to many participants",
            "Faster turnaround",
            "Lower cost per session",
            "Natural environment",
            "No facilitator bias"
        ],
        'cons': [
            "No follow-up questions",
            "Can't clarify confusion",
            "Quality varies",
            "Participant may give up"
        ],
        'best_for': [
            "Validating known patterns",
            "A/B comparisons",
            "Quick benchmarking",
            "Geographic reach"
        ],
        'tools': ['UserTesting', 'Maze', 'UsabilityHub', 'Hotjar']
    }
}
```

## Survey Design

### Question Types & When to Use

```python
class SurveyBuilder:
    """Build effective surveys"""

    QUESTION_TYPES = {
        'single_choice': {
            'use_when': "Mutually exclusive options",
            'example': "How often do you use [product]? Daily / Weekly / Monthly / Rarely",
            'analysis': "Frequency distribution, mode"
        },
        'multiple_choice': {
            'use_when': "Select all that apply",
            'example': "Which features do you use? [checkboxes]",
            'analysis': "Frequency per option"
        },
        'likert_scale': {
            'use_when': "Measuring attitudes/agreement",
            'example': "I find this product easy to use. Strongly disagree → Strongly agree",
            'analysis': "Mean, distribution, segment comparison",
            'tips': [
                "Use 5 or 7 point scales",
                "Label all points, not just endpoints",
                "Keep direction consistent",
                "Consider forced choice (no neutral)"
            ]
        },
        'ranking': {
            'use_when': "Priority/preference order",
            'example': "Rank these features by importance (1 = most important)",
            'analysis': "Average rank, top-box analysis"
        },
        'open_ended': {
            'use_when': "Exploratory, need rich detail",
            'example': "What's the biggest challenge you face with [task]?",
            'analysis': "Thematic coding, word frequency",
            'tips': [
                "Limit to 1-2 open questions",
                "Place at end of survey",
                "Make optional if possible"
            ]
        },
        'matrix': {
            'use_when': "Same scale for multiple items",
            'example': "Rate each feature: Not useful → Very useful",
            'analysis': "Per-item means, factor analysis",
            'tips': [
                "Max 5-7 rows to avoid fatigue",
                "Randomize row order",
                "Break into multiple matrices if needed"
            ]
        }
    }

    def build_survey(self, questions: List[Dict]) -> Dict:
        """
        Survey structure best practices:
        1. Start with easy, engaging questions
        2. Group related questions
        3. Place sensitive/demographic questions last
        4. Progress from general to specific
        """
        return {
            'intro': "This survey takes ~[X] minutes. Your responses are anonymous.",
            'sections': [
                {'name': 'Screening', 'questions': []},
                {'name': 'Usage', 'questions': []},
                {'name': 'Satisfaction', 'questions': []},
                {'name': 'Demographics', 'questions': []}
            ],
            'outro': "Thank you for your feedback!"
        }

    @staticmethod
    def calculate_sample_size(
        population: int,
        confidence_level: float = 0.95,
        margin_of_error: float = 0.05
    ) -> int:
        """
        Calculate required sample size

        Common targets:
        - 95% confidence, 5% margin: ~385 for large populations
        - 90% confidence, 5% margin: ~270
        """
        import scipy.stats as stats

        z = stats.norm.ppf(1 - (1 - confidence_level) / 2)
        p = 0.5  # Maximum variability assumption

        n = (z**2 * p * (1 - p)) / (margin_of_error**2)

        # Finite population correction
        if population < 10000:
            n = n / (1 + (n - 1) / population)

        return int(n) + 1


SURVEY_PITFALLS = [
    "Leading questions ('How great is our new feature?')",
    "Double-barreled questions ('How satisfied are you with speed and reliability?')",
    "Survey too long (>10 minutes = high abandonment)",
    "Required questions users can't answer",
    "Missing 'Not applicable' or 'I don't know' options",
    "Biased response scales",
    "No mobile optimization"
]
```

## Contextual Inquiry

```python
class ContextualInquiry:
    """
    Contextual Inquiry: Observing users in their natural environment

    Based on Beyer & Holtzblatt's methodology
    """

    PRINCIPLES = {
        'context': "Go to user's workplace, observe real work",
        'partnership': "Collaborate as master-apprentice",
        'interpretation': "Check understanding in real-time",
        'focus': "Stay on topic but follow interesting tangents"
    }

    def session_structure(self) -> Dict:
        return {
            'setup': {
                'duration': '15 min',
                'activities': [
                    "Explain purpose and get consent",
                    "Set up recording if allowed",
                    "Ask for tour of workspace"
                ]
            },
            'observation': {
                'duration': '60-90 min',
                'activities': [
                    "Watch user do real work",
                    "Ask about what they're doing",
                    "Note workarounds and pain points",
                    "Photograph artifacts (with permission)"
                ],
                'probes': [
                    "What are you doing now?",
                    "Why did you do it that way?",
                    "Is that typical?",
                    "What happens next?",
                    "Tell me about this [artifact]"
                ]
            },
            'wrap_up': {
                'duration': '15 min',
                'activities': [
                    "Summarize observations",
                    "Verify interpretations",
                    "Ask clarifying questions"
                ]
            }
        }

    def analysis_models(self) -> List[str]:
        """Work models from contextual design"""
        return [
            "Flow Model: Communication between people/systems",
            "Sequence Model: Step-by-step task breakdown",
            "Artifact Model: Documents/tools used",
            "Cultural Model: Policies, values, constraints",
            "Physical Model: Environment layout"
        ]
```

## Diary Studies

```python
class DiaryStudy:
    """
    Diary Study: Longitudinal self-reporting

    Participants record experiences over time
    """

    def design_study(
        self,
        duration_days: int = 14,
        entries_per_day: int = 2,
        entry_format: str = 'structured'
    ) -> Dict:
        return {
            'duration': duration_days,
            'frequency': f"{entries_per_day}x daily",
            'entry_types': {
                'structured': {
                    'fields': [
                        'What activity were you doing?',
                        'Where were you?',
                        'How did you feel? (1-5)',
                        'What tools did you use?',
                        'Any frustrations or delights?'
                    ],
                    'pros': "Easy to analyze, consistent",
                    'cons': "May miss unexpected insights"
                },
                'snippet': {
                    'fields': [
                        'Quick photo/screenshot',
                        'One sentence description',
                        'Emotion tag'
                    ],
                    'pros': "Low burden, captures moments",
                    'cons': "Less detail"
                },
                'open_ended': {
                    'fields': [
                        'Describe your experience',
                        'Optional media upload'
                    ],
                    'pros': "Rich, exploratory data",
                    'cons': "Variable quality, harder to analyze"
                }
            },
            'prompts': {
                'trigger_based': "When you [event], record an entry",
                'scheduled': "Record at 9am and 6pm daily",
                'random': "Respond when you get notification"
            },
            'retention_tactics': [
                "Daily reminders (not annoying)",
                "Progress indicators",
                "Quick entry format (<2 min)",
                "Adequate incentive",
                "Mid-study check-in call"
            ]
        }

    def analyze_entries(self, entries: List[Dict]) -> Dict:
        """Analyze diary entries for patterns"""
        return {
            'temporal_patterns': "When do events occur?",
            'location_patterns': "Where do activities happen?",
            'emotional_trends': "How do feelings change over time?",
            'trigger_analysis': "What causes positive/negative experiences?",
            'journey_reconstruction': "Timeline of key moments"
        }
```

## Card Sorting

```python
class CardSorting:
    """
    Card Sorting: Understanding mental models for IA

    Open sort: Participants create categories
    Closed sort: Participants sort into predefined categories
    Hybrid: Some predefined, can add new
    """

    def setup_study(
        self,
        cards: List[str],
        sort_type: str = 'open'
    ) -> Dict:
        return {
            'cards': cards,
            'type': sort_type,
            'instructions': {
                'open': """
                    Please group these items in a way that makes sense to you.
                    Create as many groups as you need.
                    Name each group when you're done.
                """,
                'closed': """
                    Please sort these items into the categories provided.
                    If an item doesn't fit anywhere, put it in 'Unsure'.
                """
            },
            'tools': ['OptimalSort', 'UserZoom', 'Miro', 'Physical cards']
        }

    def analyze_results(self, sorts: List[Dict]) -> Dict:
        """
        Analysis methods:
        - Similarity matrix: How often items grouped together
        - Dendrograms: Hierarchical clustering
        - Category frequency: Most common groupings
        """
        return {
            'similarity_matrix': "Items A & B grouped together by X% of participants",
            'suggested_categories': "Based on clustering",
            'problem_cards': "Items sorted inconsistently",
            'agreement_score': "Overall consistency across participants"
        }
```

## Participant Recruitment

```python
class RecruitmentPlan:
    """Plan and execute participant recruitment"""

    def create_screener(self, criteria: Dict) -> List[Dict]:
        """
        Screener survey to qualify participants

        Criteria types:
        - Demographics (age, location, occupation)
        - Behavioral (usage frequency, experience level)
        - Attitudinal (preferences, goals)
        """
        questions = []

        # Disguise qualifying criteria
        for criterion, values in criteria.items():
            questions.append({
                'question': f"Which best describes your {criterion}?",
                'options': values['options'] + values['disqualify_options'],
                'qualify': values['options']
            })

        return questions

    RECRUITMENT_CHANNELS = {
        'existing_users': {
            'methods': ['In-app invite', 'Email list', 'Support tickets'],
            'pros': "Know the product, easy to reach",
            'cons': "May be biased, power users"
        },
        'recruitment_panels': {
            'methods': ['UserTesting', 'Respondent', 'User Interviews'],
            'pros': "Fast, diverse, pre-screened",
            'cons': "Cost, professional participants"
        },
        'social_media': {
            'methods': ['LinkedIn', 'Reddit communities', 'Facebook groups'],
            'pros': "Targeted communities, authentic",
            'cons': "Slow, variable quality"
        },
        'intercepts': {
            'methods': ['Website popup', 'In-app modal'],
            'pros': "Catch users in context",
            'cons': "Interruptive, may skew to casual users"
        }
    }

    INCENTIVE_GUIDELINES = {
        '30_min_survey': '$10-25',
        '60_min_interview': '$50-100',
        '90_min_usability': '$75-150',
        'diary_study_week': '$100-200',
        'contextual_inquiry': '$150-300',
        'b2b_professional': '2-3x consumer rates'
    }
```

## Best Practices

1. **Triangulate**: Combine multiple methods to validate findings
2. **Start with why**: Define research questions before choosing methods
3. **5 users finds 85% of issues**: For usability testing (Nielsen)
4. **Separate behavior from preference**: What people do ≠ what they say
5. **Involve stakeholders**: Share research process, not just results

## Common Pitfalls

- **Confirmation bias**: Seeking data that supports existing beliefs
- **Leading questions**: Influencing participant responses
- **Sampling bias**: Recruiting only easily available participants
- **Over-relying on surveys**: Missing qualitative context
- **Not recording**: Relying on memory and notes alone
- **Analysis paralysis**: Collecting more data than can be synthesized

---

**Skill Type**: UX - User Research
**Complexity**: Intermediate to Advanced
**Typical Usage**: Discovery phase, design validation, continuous research
