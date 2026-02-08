---
name: persona-journey-mapping
description: Load when user needs persona development, empathy maps, user journey maps, service blueprints, or experience mapping. Covers user modeling and journey visualization.
trigger_keywords: [persona, user persona, empathy map, journey map, customer journey, user journey, service blueprint, experience map, jobs to be done, jtbd, user stories, scenario mapping, touchpoints]
---

# Persona & Journey Mapping Skill

Techniques for modeling users and visualizing their experiences across touchpoints.

## Core Concepts

- **Research-Backed Personas**: Composite user archetypes derived from actual research data (interviews, surveys), not assumptions - include quotes and data sources for credibility
- **Jobs To Be Done (JTBD)**: Focus on functional, emotional, and social jobs users hire products for rather than demographic attributes - reveals true motivations
- **Empathy Mapping**: Visual framework capturing what users Say, Think, Do, and Feel - synthesizes qualitative research into actionable insights about user mindset
- **Journey Map Anatomy**: Visualization of user experience across stages (awareness to advocacy), documenting actions, thoughts, emotions, pain points, and opportunities at each touchpoint
- **Service Blueprinting**: Extended journey map adding frontstage/backstage organizational processes - reveals how internal operations affect customer experience

## Persona Development

### Research-Based Personas

```python
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum

@dataclass
class Persona:
    """
    Research-backed persona template

    Personas must be based on research, not assumptions.
    Include data sources and quotes for credibility.
    """

    # Identity
    name: str  # Memorable, realistic name
    archetype: str  # Role label ("The Busy Professional")
    photo: str  # Realistic stock photo path

    # Demographics
    age_range: str  # "35-45"
    occupation: str
    location: str
    income_range: Optional[str] = None

    # Behavioral attributes (from research)
    goals: List[str] = field(default_factory=list)
    frustrations: List[str] = field(default_factory=list)
    motivations: List[str] = field(default_factory=list)

    # Context
    tech_savviness: str = "Medium"  # Low/Medium/High
    tools_used: List[str] = field(default_factory=list)
    typical_day: str = ""

    # Quotes from research
    verbatims: List[str] = field(default_factory=list)

    # Data source
    based_on: str = ""  # "8 user interviews, 150 survey responses"

    def to_card(self) -> Dict:
        """Export as persona card format"""
        return {
            'header': {
                'name': self.name,
                'archetype': self.archetype,
                'photo': self.photo,
                'tagline': self.verbatims[0] if self.verbatims else ""
            },
            'demographics': {
                'age': self.age_range,
                'occupation': self.occupation,
                'location': self.location
            },
            'psychographics': {
                'goals': self.goals,
                'frustrations': self.frustrations,
                'motivations': self.motivations
            },
            'context': {
                'tech_savviness': self.tech_savviness,
                'tools': self.tools_used,
                'typical_day': self.typical_day
            },
            'data_source': self.based_on
        }


# Example persona
marketing_manager = Persona(
    name="Sarah Chen",
    archetype="The Data-Driven Marketer",
    photo="sarah_chen.jpg",
    age_range="32-40",
    occupation="Marketing Manager",
    location="Urban, US",
    income_range="$80K-$120K",
    goals=[
        "Prove marketing ROI to leadership",
        "Automate repetitive reporting tasks",
        "Stay ahead of industry trends"
    ],
    frustrations=[
        "Data scattered across too many tools",
        "Reports take hours to compile",
        "Can't get real-time campaign insights"
    ],
    motivations=[
        "Career advancement",
        "Being seen as innovative",
        "Work-life balance"
    ],
    tech_savviness="High",
    tools_used=["Google Analytics", "HubSpot", "Tableau", "Slack"],
    typical_day="Mornings: Check campaign metrics. Afternoons: Meetings and strategy. Evenings: Catch up on industry news.",
    verbatims=[
        "I spend more time making reports than acting on insights.",
        "If I could get real-time data, I could make faster decisions."
    ],
    based_on="12 user interviews, 200 survey responses (Q3 2024)"
)
```

### Persona Spectrum

```python
class PersonaSpectrum:
    """
    Alternative to discrete personas: behavioral spectrums

    Shows range of user behaviors rather than fixed types
    """

    def define_spectrums(self) -> List[Dict]:
        """
        Spectrums represent behavioral dimensions
        Users fall somewhere on each spectrum
        """
        return [
            {
                'dimension': 'Tech Adoption',
                'low_end': 'Late Majority',
                'high_end': 'Early Adopter',
                'description': "How quickly they adopt new technology"
            },
            {
                'dimension': 'Decision Style',
                'low_end': 'Methodical Researcher',
                'high_end': 'Quick Decider',
                'description': "How much research before action"
            },
            {
                'dimension': 'Support Preference',
                'low_end': 'Self-Service',
                'high_end': 'High-Touch',
                'description': "Preferred level of human support"
            },
            {
                'dimension': 'Risk Tolerance',
                'low_end': 'Risk-Averse',
                'high_end': 'Risk-Tolerant',
                'description': "Willingness to try unproven solutions"
            }
        ]

    def plot_user_on_spectrum(
        self,
        user_id: str,
        scores: Dict[str, float]  # dimension: 0-100 score
    ) -> Dict:
        """Place a user segment on spectrums"""
        return {
            'user_id': user_id,
            'positions': {
                dim: {'score': score, 'label': self._score_to_label(score)}
                for dim, score in scores.items()
            }
        }

    def _score_to_label(self, score: float) -> str:
        if score < 25:
            return "Strongly Low"
        elif score < 50:
            return "Somewhat Low"
        elif score < 75:
            return "Somewhat High"
        else:
            return "Strongly High"
```

### Jobs To Be Done (JTBD)

```python
@dataclass
class JobToBeDone:
    """
    Jobs To Be Done framework

    Focus on what users are trying to accomplish,
    not who they are or what features they want.

    Format: "When [situation], I want to [motivation],
            so I can [outcome]"
    """

    situation: str  # Context/trigger
    motivation: str  # What they want to do
    outcome: str  # Desired end state

    # Job dimensions
    functional: str = ""  # Practical task
    emotional: str = ""  # How they want to feel
    social: str = ""  # How they want to be perceived

    # Forces
    push_current: List[str] = field(default_factory=list)  # Problems with current solution
    pull_new: List[str] = field(default_factory=list)  # Attraction to new solution
    anxiety_new: List[str] = field(default_factory=list)  # Concerns about switching
    habit_current: List[str] = field(default_factory=list)  # Comfort with status quo

    def job_statement(self) -> str:
        return f"When {self.situation}, I want to {self.motivation}, so I can {self.outcome}"

    def forces_diagram(self) -> Dict:
        """
        Forces of Progress diagram

        Progress happens when Pull + Push > Anxiety + Habit
        """
        return {
            'promoting_progress': {
                'push_of_current': self.push_current,
                'pull_of_new': self.pull_new
            },
            'opposing_progress': {
                'anxiety_of_new': self.anxiety_new,
                'habit_of_current': self.habit_current
            }
        }


# Example JTBD
onboarding_job = JobToBeDone(
    situation="I start using a new project management tool",
    motivation="quickly understand how to use it for my workflow",
    outcome="feel confident and productive within the first day",
    functional="Set up my first project with tasks and team members",
    emotional="Feel competent, not overwhelmed",
    social="Show my team I made a good tool choice",
    push_current=[
        "Current tool is slow and clunky",
        "Team complaining about missed deadlines"
    ],
    pull_new=[
        "Heard great reviews",
        "Free trial available",
        "Integrates with tools we already use"
    ],
    anxiety_new=[
        "Learning curve might slow us down",
        "What if team doesn't adopt it?",
        "Migration might lose data"
    ],
    habit_current=[
        "Know all the workarounds",
        "Templates already set up",
        "Don't have time to learn something new"
    ]
)
```

## Empathy Maps

```python
@dataclass
class EmpathyMap:
    """
    Empathy Map: Understanding user's perspective

    Four quadrants + center (user goals, pains, gains)
    """

    persona_name: str

    # Four quadrants
    says: List[str] = field(default_factory=list)  # Direct quotes
    thinks: List[str] = field(default_factory=list)  # Beliefs, assumptions
    does: List[str] = field(default_factory=list)  # Actions, behaviors
    feels: List[str] = field(default_factory=list)  # Emotions

    # Center
    goals: List[str] = field(default_factory=list)  # What they want to achieve
    pains: List[str] = field(default_factory=list)  # Frustrations, obstacles
    gains: List[str] = field(default_factory=list)  # Benefits they seek

    def validate(self) -> List[str]:
        """Check empathy map quality"""
        issues = []

        if len(self.says) < 3:
            issues.append("Need more direct quotes (Says)")
        if len(self.does) < 3:
            issues.append("Need more observed behaviors (Does)")
        if any(s in self.thinks for s in self.says):
            issues.append("Thinks should be internal; Says is external")
        if not self.goals:
            issues.append("Missing user goals")

        return issues

    def to_canvas(self) -> Dict:
        """Export as canvas format for visualization"""
        return {
            'title': f"Empathy Map: {self.persona_name}",
            'quadrants': {
                'top_left': {'label': 'SAYS', 'items': self.says},
                'top_right': {'label': 'THINKS', 'items': self.thinks},
                'bottom_left': {'label': 'DOES', 'items': self.does},
                'bottom_right': {'label': 'FEELS', 'items': self.feels}
            },
            'center': {
                'goals': self.goals,
                'pains': self.pains,
                'gains': self.gains
            }
        }


# Example empathy map
sarah_empathy = EmpathyMap(
    persona_name="Sarah Chen",
    says=[
        "I don't have time to learn a new tool",
        "Can you just send me the numbers?",
        "I need this report by end of day"
    ],
    thinks=[
        "Am I measuring the right things?",
        "Leadership doesn't understand marketing",
        "There has to be a better way to do this"
    ],
    does=[
        "Checks email first thing every morning",
        "Exports data to Excel for analysis",
        "Screenshots charts for presentations",
        "Works late to finish reports"
    ],
    feels=[
        "Frustrated with manual processes",
        "Anxious about proving ROI",
        "Overwhelmed by data volume",
        "Proud when campaigns succeed"
    ],
    goals=[
        "Reduce time spent on reporting",
        "Make data-driven decisions faster",
        "Gain respect from leadership"
    ],
    pains=[
        "Data silos across tools",
        "Manual copy-paste workflows",
        "Stakeholder requests at last minute"
    ],
    gains=[
        "Real-time dashboards",
        "Automated reports",
        "Clear attribution models"
    ]
)
```

## User Journey Maps

### Journey Map Structure

```python
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum

class EmotionLevel(Enum):
    DELIGHTED = 2
    HAPPY = 1
    NEUTRAL = 0
    FRUSTRATED = -1
    ANGRY = -2

@dataclass
class Touchpoint:
    """Single interaction point in journey"""

    name: str
    channel: str  # Web, Mobile, Email, Phone, In-person
    action: str  # What user does
    thinking: str  # What user thinks
    feeling: EmotionLevel
    pain_points: List[str] = field(default_factory=list)
    opportunities: List[str] = field(default_factory=list)

@dataclass
class JourneyStage:
    """Major phase of the journey"""

    name: str
    goal: str  # User's goal in this stage
    touchpoints: List[Touchpoint] = field(default_factory=list)
    duration: str = ""  # "5 minutes", "2-3 days"

@dataclass
class UserJourneyMap:
    """
    Complete user journey visualization

    Shows the end-to-end experience across touchpoints
    """

    title: str
    persona: str
    scenario: str  # Specific use case
    stages: List[JourneyStage] = field(default_factory=list)

    # Swim lanes (horizontal layers)
    include_emotions: bool = True
    include_channels: bool = True
    include_opportunities: bool = True
    include_backstage: bool = False  # Internal processes

    def add_stage(self, stage: JourneyStage):
        self.stages.append(stage)

    def emotion_curve(self) -> List[Dict]:
        """Extract emotional journey for visualization"""
        curve = []
        for stage in self.stages:
            for tp in stage.touchpoints:
                curve.append({
                    'stage': stage.name,
                    'touchpoint': tp.name,
                    'emotion': tp.feeling.value,
                    'label': tp.feeling.name
                })
        return curve

    def identify_pain_points(self) -> List[Dict]:
        """Find all pain points across journey"""
        pains = []
        for stage in self.stages:
            for tp in stage.touchpoints:
                if tp.feeling.value < 0 or tp.pain_points:
                    pains.append({
                        'stage': stage.name,
                        'touchpoint': tp.name,
                        'emotion': tp.feeling.name,
                        'pain_points': tp.pain_points
                    })
        return pains

    def identify_opportunities(self) -> List[Dict]:
        """Find all opportunity areas"""
        opps = []
        for stage in self.stages:
            for tp in stage.touchpoints:
                if tp.opportunities:
                    opps.append({
                        'stage': stage.name,
                        'touchpoint': tp.name,
                        'opportunities': tp.opportunities
                    })
        return opps

    def to_mermaid(self) -> str:
        """Export as Mermaid diagram"""
        lines = ["journey", f"    title {self.title}"]
        for stage in self.stages:
            lines.append(f"    section {stage.name}")
            for tp in stage.touchpoints:
                # Mermaid journey uses 1-5 scale
                score = tp.feeling.value + 3  # Convert -2..2 to 1..5
                lines.append(f"        {tp.name}: {score}: {tp.channel}")
        return "\n".join(lines)


# Example journey map
ecommerce_journey = UserJourneyMap(
    title="First Purchase Journey",
    persona="Sarah Chen",
    scenario="Buying a laptop for work"
)

# Stage 1: Awareness
awareness = JourneyStage(
    name="Awareness",
    goal="Realize I need a new laptop",
    duration="Days to weeks",
    touchpoints=[
        Touchpoint(
            name="Laptop slows down",
            channel="In-person",
            action="Experiences frustration with current device",
            thinking="I really need to replace this",
            feeling=EmotionLevel.FRUSTRATED,
            pain_points=["Lost productivity"],
            opportunities=[]
        ),
        Touchpoint(
            name="Sees colleague's laptop",
            channel="In-person",
            action="Asks about their laptop",
            thinking="That looks fast and lightweight",
            feeling=EmotionLevel.NEUTRAL,
            pain_points=[],
            opportunities=["Word of mouth referral program"]
        )
    ]
)

# Stage 2: Research
research = JourneyStage(
    name="Research",
    goal="Find the right laptop",
    duration="1-2 weeks",
    touchpoints=[
        Touchpoint(
            name="Google search",
            channel="Web",
            action="Searches 'best laptops for business 2024'",
            thinking="So many options, where do I start?",
            feeling=EmotionLevel.NEUTRAL,
            pain_points=["Information overload"],
            opportunities=["SEO for comparison content"]
        ),
        Touchpoint(
            name="Reads reviews",
            channel="Web",
            action="Checks multiple review sites",
            thinking="Which reviews can I trust?",
            feeling=EmotionLevel.FRUSTRATED,
            pain_points=["Conflicting reviews", "Sponsored content unclear"],
            opportunities=["Transparent review sourcing"]
        ),
        Touchpoint(
            name="Visits brand website",
            channel="Web",
            action="Explores product pages",
            thinking="This looks good but is it worth the price?",
            feeling=EmotionLevel.HAPPY,
            pain_points=["Hard to compare models"],
            opportunities=["Comparison tool", "Use case recommendations"]
        )
    ]
)

# Stage 3: Purchase
purchase = JourneyStage(
    name="Purchase",
    goal="Buy the laptop",
    duration="30 minutes",
    touchpoints=[
        Touchpoint(
            name="Add to cart",
            channel="Web",
            action="Configures and adds laptop",
            thinking="Did I pick the right specs?",
            feeling=EmotionLevel.NEUTRAL,
            pain_points=["Unsure about configuration"],
            opportunities=["Configuration advisor"]
        ),
        Touchpoint(
            name="Checkout",
            channel="Web",
            action="Enters payment and shipping",
            thinking="Hope this arrives on time",
            feeling=EmotionLevel.HAPPY,
            pain_points=["Shipping cost surprise"],
            opportunities=["Free shipping threshold"]
        ),
        Touchpoint(
            name="Confirmation email",
            channel="Email",
            action="Receives order confirmation",
            thinking="Good, it went through",
            feeling=EmotionLevel.DELIGHTED,
            pain_points=[],
            opportunities=["Onboarding content", "Accessory recommendations"]
        )
    ]
)

ecommerce_journey.add_stage(awareness)
ecommerce_journey.add_stage(research)
ecommerce_journey.add_stage(purchase)
```

### Journey Map Templates

```python
JOURNEY_TEMPLATES = {
    'acquisition': {
        'stages': ['Awareness', 'Consideration', 'Decision', 'Purchase'],
        'focus': "Converting prospects to customers",
        'key_metrics': ['Conversion rate', 'Time to convert', 'Drop-off points']
    },
    'onboarding': {
        'stages': ['Sign Up', 'First Use', 'Activation', 'Habit Formation'],
        'focus': "New user success",
        'key_metrics': ['Activation rate', 'Time to value', 'Day 7 retention']
    },
    'support': {
        'stages': ['Issue Occurs', 'Seek Help', 'Resolution', 'Follow-up'],
        'focus': "Problem resolution experience",
        'key_metrics': ['Resolution time', 'CSAT', 'Repeat contacts']
    },
    'renewal': {
        'stages': ['Usage', 'Evaluation', 'Decision', 'Renewal/Churn'],
        'focus': "Retention and loyalty",
        'key_metrics': ['NPS', 'Churn rate', 'Expansion revenue']
    }
}
```

## Service Blueprints

```python
@dataclass
class ServiceBlueprint:
    """
    Service Blueprint: Journey map + behind-the-scenes

    Shows frontstage (visible to customer) and
    backstage (internal processes) together
    """

    title: str
    service: str

    # Layers (swim lanes)
    @dataclass
    class BlueprintLayer:
        physical_evidence: List[str]  # Tangible artifacts
        customer_actions: List[str]  # What customer does
        frontstage: List[str]  # Visible employee actions
        backstage: List[str]  # Invisible employee actions
        support_processes: List[str]  # Systems, partners

    stages: Dict[str, BlueprintLayer] = field(default_factory=dict)

    # Lines
    line_of_interaction: str = "Between customer and frontstage"
    line_of_visibility: str = "Between frontstage and backstage"
    line_of_internal_interaction: str = "Between backstage and support"

    def add_stage(self, name: str, layer: 'ServiceBlueprint.BlueprintLayer'):
        self.stages[name] = layer

    def identify_failure_points(self) -> List[str]:
        """Find potential failure points"""
        failures = []
        for stage, layer in self.stages.items():
            # Complex backstage processes
            if len(layer.backstage) > 3:
                failures.append(f"{stage}: Complex backstage process")
            # Customer actions without support
            if layer.customer_actions and not layer.frontstage:
                failures.append(f"{stage}: Customer action without employee support")
        return failures


# Example: Restaurant service blueprint
restaurant_blueprint = ServiceBlueprint(
    title="Restaurant Dining Experience",
    service="Fine Dining Restaurant"
)

restaurant_blueprint.add_stage(
    "Reservation",
    ServiceBlueprint.BlueprintLayer(
        physical_evidence=["Website", "Confirmation email"],
        customer_actions=["Search for restaurant", "Book online"],
        frontstage=["Automated booking confirmation"],
        backstage=["Reservation system updates availability", "Host notified"],
        support_processes=["Reservation management system", "Email service"]
    )
)

restaurant_blueprint.add_stage(
    "Arrival",
    ServiceBlueprint.BlueprintLayer(
        physical_evidence=["Exterior signage", "Host stand", "Waiting area"],
        customer_actions=["Enter restaurant", "Check in with host"],
        frontstage=["Host greets", "Confirms reservation", "Seats guest"],
        backstage=["Table prepared", "Server assigned"],
        support_processes=["Table management system", "Cleaning protocols"]
    )
)

restaurant_blueprint.add_stage(
    "Ordering",
    ServiceBlueprint.BlueprintLayer(
        physical_evidence=["Menu", "Table setting", "Server uniform"],
        customer_actions=["Review menu", "Ask questions", "Place order"],
        frontstage=["Server presents menu", "Explains specials", "Takes order"],
        backstage=["Order entered in POS", "Kitchen receives ticket"],
        support_processes=["POS system", "Inventory management"]
    )
)
```

## Experience Maps

```python
@dataclass
class ExperienceMap:
    """
    Experience Map: Broader than journey map

    Maps general human behavior not tied to specific product
    Used for exploring problem space before product exists
    """

    title: str
    behavior: str  # General behavior being mapped
    persona: Optional[str] = None  # May not have specific persona

    phases: List[Dict] = field(default_factory=list)

    def add_phase(
        self,
        name: str,
        actions: List[str],
        thoughts: List[str],
        feelings: List[str],
        influences: List[str],
        tools_used: List[str]
    ):
        self.phases.append({
            'name': name,
            'actions': actions,
            'thoughts': thoughts,
            'feelings': feelings,
            'influences': influences,  # What affects decisions
            'tools_used': tools_used  # Current solutions
        })

    def identify_gaps(self) -> List[Dict]:
        """Find unmet needs and gaps in current tools"""
        gaps = []
        for phase in self.phases:
            if 'workaround' in ' '.join(phase['actions']).lower():
                gaps.append({'phase': phase['name'], 'type': 'workaround'})
            if 'frustrated' in ' '.join(phase['feelings']).lower():
                gaps.append({'phase': phase['name'], 'type': 'frustration'})
        return gaps


# Example: General "Managing personal finances" experience
finance_experience = ExperienceMap(
    title="Managing Personal Finances",
    behavior="How people track and plan their money"
)

finance_experience.add_phase(
    name="Tracking Spending",
    actions=[
        "Check bank account",
        "Review credit card statements",
        "Categorize transactions",
        "Export to spreadsheet (workaround)"
    ],
    thoughts=[
        "Where did all my money go?",
        "I should track this better",
        "These categories don't match my needs"
    ],
    feelings=["Overwhelmed", "Guilty", "Frustrated with manual work"],
    influences=["Bank notifications", "End of month statements"],
    tools_used=["Bank app", "Excel/Google Sheets", "Mint", "Paper receipts"]
)
```

## Scenario Mapping

```python
@dataclass
class Scenario:
    """
    User Scenario: Narrative describing user interaction

    More detailed than user story, shows full context
    """

    persona: str
    context: str  # Situation/environment
    trigger: str  # What prompts the action
    goal: str  # What they want to achieve
    narrative: str  # Full story
    success_criteria: List[str]  # How we know they succeeded

    def to_user_story(self) -> str:
        """Convert to user story format"""
        return f"As {self.persona}, I want to {self.goal}, so that {self.success_criteria[0]}"

    def to_acceptance_criteria(self) -> List[str]:
        """Convert to Given-When-Then format"""
        return [
            f"Given {self.context}",
            f"When {self.trigger}",
            f"Then {criterion}"
            for criterion in self.success_criteria
        ]


# Example scenario
budget_scenario = Scenario(
    persona="Sarah Chen",
    context="End of month, sitting at home laptop, has 30 minutes before dinner",
    trigger="Receives notification that she's close to credit limit",
    goal="understand where money went this month and make a plan",
    narrative="""
    Sarah just got home from work and received an alert that her credit card
    is close to its limit. Surprised, she opens her banking app to understand
    what happened. She sees several large transactions but can't remember what
    they were for. She exports the data to a spreadsheet and spends 20 minutes
    categorizing expenses. She realizes she overspent on dining out and
    subscriptions she forgot about. She makes a mental note to cancel some
    subscriptions but doesn't have time now. She feels frustrated that this
    happens every month and wishes there was an easier way.
    """,
    success_criteria=[
        "Can quickly see spending by category",
        "Can identify unusual or forgotten subscriptions",
        "Can set a budget for next month",
        "Feels in control of finances"
    ]
)
```

## Best Practices

1. **Base on research**: Personas and journeys without research are fiction
2. **Keep it focused**: One persona, one journey, one scenario at a time
3. **Make it visual**: Use diagrams, not just text
4. **Include emotions**: Emotional curve reveals opportunity areas
5. **Update regularly**: Journeys change as product and market evolve
6. **Share widely**: Personas/journeys should be visible to entire team

## Common Pitfalls

- **Persona proliferation**: Too many personas dilute focus (3-5 max)
- **Demographic-only personas**: Focus on behaviors, not just demographics
- **Aspirational journeys**: Map reality, not ideal state
- **Too much detail**: Keep maps scannable, details in appendix
- **No stakeholder buy-in**: Involve team in creation for adoption
- **One-time exercise**: Journey mapping is ongoing, not one-time

---

**Skill Type**: UX - Persona & Journey Mapping
**Complexity**: Intermediate
**Typical Usage**: Discovery, strategy, team alignment
