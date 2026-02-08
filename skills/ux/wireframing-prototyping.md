---
name: wireframing-prototyping
description: Load when user needs wireframing techniques, prototyping patterns, design tool workflows, Figma API integration, or interaction design. Covers fidelity levels and rapid iteration.
trigger_keywords: [wireframe, prototype, mockup, figma, sketch, low fidelity, high fidelity, interaction design, clickable prototype, design handoff, design specs, responsive design, mobile design]
---

# Wireframing & Prototyping Skill

Creating wireframes and prototypes at various fidelity levels for design exploration and validation.

## Core Concepts

- **Fidelity Spectrum**: Sketch (minutes, ideation) to Low-fi (grayscale boxes) to Mid-fi (basic styling) to High-fi (pixel-perfect) - match fidelity to design phase and testing goals
- **Prototype vs. Mockup**: Prototypes are interactive (clickable, demonstrate behavior); mockups are static (visual design only) - choose based on what you need to validate
- **Appropriate Fidelity Principle**: Use lowest fidelity that answers your question; high-fidelity too early wastes time and biases feedback toward visual details over interaction patterns
- **Rapid Iteration**: Wireframes enable fast exploration of layout and hierarchy; prototype multiple concepts before committing - it's cheaper to iterate on paper than code
- **Design Handoff Specs**: Final deliverables include component specifications, interaction states, responsive breakpoints, and annotation of spacing/typography for developer implementation

## Fidelity Spectrum

### Understanding Fidelity Levels

```python
from dataclasses import dataclass
from typing import List, Dict
from enum import Enum

class FidelityLevel(Enum):
    SKETCH = "sketch"  # Paper/whiteboard
    LOW = "low"  # Grayscale boxes
    MID = "mid"  # Basic styling, placeholder content
    HIGH = "high"  # Pixel-perfect, real content

@dataclass
class FidelityGuidelines:
    """When to use each fidelity level"""

    level: FidelityLevel
    time_to_create: str
    use_when: List[str]
    avoid_when: List[str]
    tools: List[str]
    deliverable: str

FIDELITY_GUIDE = {
    FidelityLevel.SKETCH: FidelityGuidelines(
        level=FidelityLevel.SKETCH,
        time_to_create="Minutes",
        use_when=[
            "Initial ideation",
            "Exploring many concepts quickly",
            "Collaborative brainstorming",
            "Early stakeholder alignment"
        ],
        avoid_when=[
            "Need to test specific interactions",
            "Stakeholders expect polished output",
            "Documenting final designs"
        ],
        tools=["Paper", "Whiteboard", "Excalidraw", "FigJam"],
        deliverable="Photos or digital captures"
    ),
    FidelityLevel.LOW: FidelityGuidelines(
        level=FidelityLevel.LOW,
        time_to_create="Hours",
        use_when=[
            "Testing information architecture",
            "Validating content hierarchy",
            "Quick iteration cycles",
            "Focus groups on structure"
        ],
        avoid_when=[
            "Testing visual design",
            "Usability testing with real tasks",
            "Developer handoff"
        ],
        tools=["Balsamiq", "Whimsical", "Figma (wireframe kit)"],
        deliverable="Grayscale screens, basic annotations"
    ),
    FidelityLevel.MID: FidelityGuidelines(
        level=FidelityLevel.MID,
        time_to_create="Days",
        use_when=[
            "Usability testing",
            "Stakeholder reviews",
            "Developer discussions",
            "Iterating on specific flows"
        ],
        avoid_when=[
            "Brand presentation",
            "Final design sign-off",
            "Marketing materials"
        ],
        tools=["Figma", "Sketch", "Adobe XD"],
        deliverable="Styled screens with placeholder content"
    ),
    FidelityLevel.HIGH: FidelityGuidelines(
        level=FidelityLevel.HIGH,
        time_to_create="Weeks",
        use_when=[
            "Final design approval",
            "Developer handoff",
            "User acceptance testing",
            "Marketing/sales demos"
        ],
        avoid_when=[
            "Early exploration",
            "Rapid iteration needed",
            "Requirements still changing"
        ],
        tools=["Figma", "Sketch", "Framer"],
        deliverable="Pixel-perfect designs, prototypes, specs"
    )
}

def recommend_fidelity(
    project_stage: str,
    time_available: str,
    testing_needed: bool,
    stakeholder_expectations: str
) -> FidelityLevel:
    """Recommend appropriate fidelity level"""
    if project_stage == 'discovery' and time_available == 'short':
        return FidelityLevel.SKETCH
    elif testing_needed and project_stage in ['discovery', 'design']:
        return FidelityLevel.MID
    elif stakeholder_expectations == 'polished' or project_stage == 'handoff':
        return FidelityLevel.HIGH
    else:
        return FidelityLevel.LOW
```

## Wireframing Patterns

### Common UI Patterns

```python
WIREFRAME_PATTERNS = {
    'navigation': {
        'top_nav': {
            'description': "Horizontal navigation at top",
            'best_for': "5-7 primary sections",
            'sketch': """
            ┌────────────────────────────────────┐
            │ Logo   Nav1  Nav2  Nav3   [Search] │
            └────────────────────────────────────┘
            """
        },
        'side_nav': {
            'description': "Vertical navigation on left",
            'best_for': "Many sections, dashboard apps",
            'sketch': """
            ┌────┬───────────────────────┐
            │Logo│                       │
            │────│                       │
            │Nav1│     Content Area      │
            │Nav2│                       │
            │Nav3│                       │
            └────┴───────────────────────┘
            """
        },
        'hamburger': {
            'description': "Hidden menu behind icon",
            'best_for': "Mobile, secondary nav",
            'sketch': """
            ┌────────────────────────────────────┐
            │ ☰   Logo                   [User]  │
            └────────────────────────────────────┘
            """
        }
    },
    'content': {
        'card_grid': {
            'description': "Grid of content cards",
            'best_for': "Products, articles, media",
            'sketch': """
            ┌─────────┐ ┌─────────┐ ┌─────────┐
            │ [Image] │ │ [Image] │ │ [Image] │
            │ Title   │ │ Title   │ │ Title   │
            │ Desc... │ │ Desc... │ │ Desc... │
            └─────────┘ └─────────┘ └─────────┘
            """
        },
        'list_view': {
            'description': "Vertical list of items",
            'best_for': "Search results, data tables",
            'sketch': """
            ┌────────────────────────────────────┐
            │ [img] Title        Meta    Action  │
            ├────────────────────────────────────┤
            │ [img] Title        Meta    Action  │
            ├────────────────────────────────────┤
            │ [img] Title        Meta    Action  │
            └────────────────────────────────────┘
            """
        },
        'hero': {
            'description': "Large featured content area",
            'best_for': "Landing pages, marketing",
            'sketch': """
            ┌────────────────────────────────────┐
            │                                    │
            │         Large Hero Image           │
            │     Headline Text                  │
            │     Supporting copy                │
            │         [ CTA Button ]             │
            │                                    │
            └────────────────────────────────────┘
            """
        }
    },
    'forms': {
        'single_column': {
            'description': "Fields stacked vertically",
            'best_for': "Simple forms, mobile",
            'sketch': """
            ┌────────────────────────────────────┐
            │ Label                              │
            │ [________________]                 │
            │                                    │
            │ Label                              │
            │ [________________]                 │
            │                                    │
            │ Label                              │
            │ [________________]                 │
            │                                    │
            │     [ Submit ]                     │
            └────────────────────────────────────┘
            """
        },
        'multi_step': {
            'description': "Wizard-style progression",
            'best_for': "Complex forms, onboarding",
            'sketch': """
            ┌────────────────────────────────────┐
            │    (1)───(2)───(3)───(4)           │
            │     ●─────○─────○─────○            │
            │                                    │
            │         Step 1 Content             │
            │                                    │
            │   [ Back ]        [ Next ]         │
            └────────────────────────────────────┘
            """
        }
    }
}

def get_pattern_template(category: str, pattern: str) -> str:
    """Get ASCII wireframe template"""
    return WIREFRAME_PATTERNS.get(category, {}).get(pattern, {}).get('sketch', '')
```

### Layout Systems

```python
LAYOUT_SYSTEMS = {
    '12_column_grid': {
        'description': "Standard responsive grid",
        'breakpoints': {
            'mobile': '< 768px (stack columns)',
            'tablet': '768px - 1024px (8 columns)',
            'desktop': '> 1024px (12 columns)'
        },
        'common_layouts': {
            'full_width': '12 cols',
            'sidebar_content': '3 + 9 cols',
            'three_column': '4 + 4 + 4 cols',
            'content_centered': '2 + 8 + 2 cols'
        }
    },
    'spacing_scale': {
        'description': "Consistent spacing multiples",
        'base': 8,  # 8px base unit
        'scale': {
            'xs': '4px (0.5x)',
            'sm': '8px (1x)',
            'md': '16px (2x)',
            'lg': '24px (3x)',
            'xl': '32px (4x)',
            'xxl': '48px (6x)'
        }
    }
}

class WireframeLayout:
    """Generate layout specifications"""

    def __init__(self, columns: int = 12, gutter: int = 24, margin: int = 24):
        self.columns = columns
        self.gutter = gutter
        self.margin = margin

    def calculate_column_width(self, container_width: int) -> float:
        """Calculate single column width"""
        available = container_width - (2 * self.margin) - ((self.columns - 1) * self.gutter)
        return available / self.columns

    def span_width(self, span: int, container_width: int) -> float:
        """Calculate width for column span"""
        col_width = self.calculate_column_width(container_width)
        return (col_width * span) + (self.gutter * (span - 1))

    def responsive_breakpoints(self) -> Dict:
        return {
            'mobile': {'width': 375, 'columns': 4, 'gutter': 16, 'margin': 16},
            'tablet': {'width': 768, 'columns': 8, 'gutter': 20, 'margin': 32},
            'desktop': {'width': 1440, 'columns': 12, 'gutter': 24, 'margin': 80}
        }
```

## Prototyping Techniques

### Interaction Patterns

```python
@dataclass
class Interaction:
    """Define an interaction for prototyping"""

    trigger: str  # click, hover, scroll, time
    source: str  # Element that triggers
    action: str  # navigate, overlay, swap, scroll-to
    target: str  # Destination or target element
    animation: str = "dissolve"  # dissolve, slide, push, instant
    duration_ms: int = 300

COMMON_INTERACTIONS = {
    'navigation': [
        Interaction(
            trigger='click',
            source='nav_item',
            action='navigate',
            target='destination_page',
            animation='dissolve'
        ),
        Interaction(
            trigger='click',
            source='back_button',
            action='navigate',
            target='previous_page',
            animation='slide_right'
        )
    ],
    'modals': [
        Interaction(
            trigger='click',
            source='open_modal_button',
            action='overlay',
            target='modal_frame',
            animation='dissolve'
        ),
        Interaction(
            trigger='click',
            source='modal_close',
            action='close_overlay',
            target='modal_frame',
            animation='dissolve'
        ),
        Interaction(
            trigger='click',
            source='modal_backdrop',
            action='close_overlay',
            target='modal_frame',
            animation='dissolve'
        )
    ],
    'dropdowns': [
        Interaction(
            trigger='click',
            source='dropdown_trigger',
            action='open_overlay',
            target='dropdown_menu',
            animation='instant'
        ),
        Interaction(
            trigger='click',
            source='dropdown_item',
            action='swap',
            target='selected_value',
            animation='instant'
        )
    ],
    'tabs': [
        Interaction(
            trigger='click',
            source='tab_1',
            action='swap',
            target='tab_content_1',
            animation='instant'
        )
    ],
    'scroll_effects': [
        Interaction(
            trigger='scroll',
            source='page',
            action='sticky',
            target='header',
            animation='instant'
        ),
        Interaction(
            trigger='scroll_to',
            source='nav_link',
            action='scroll_to',
            target='section_anchor',
            animation='smooth'
        )
    ]
}
```

### Micro-interactions

```python
MICRO_INTERACTIONS = {
    'feedback': {
        'button_press': {
            'trigger': 'On press',
            'animation': 'Scale down to 95%, darken 10%',
            'duration': '100ms'
        },
        'form_success': {
            'trigger': 'On submit success',
            'animation': 'Green checkmark appears, form fades',
            'duration': '300ms'
        },
        'error_shake': {
            'trigger': 'On validation error',
            'animation': 'Input shakes horizontally 3 times',
            'duration': '300ms'
        }
    },
    'loading': {
        'skeleton': {
            'description': 'Placeholder shapes during load',
            'animation': 'Shimmer effect left to right',
            'when': 'Content loading, known layout'
        },
        'spinner': {
            'description': 'Rotating indicator',
            'animation': 'Continuous rotation',
            'when': 'Unknown content, short wait'
        },
        'progress_bar': {
            'description': 'Linear progress indicator',
            'animation': 'Width increases with progress',
            'when': 'Known progress percentage'
        }
    },
    'transitions': {
        'page_enter': {
            'animation': 'Fade in from 0 opacity, slide up 20px',
            'duration': '300ms',
            'easing': 'ease-out'
        },
        'list_item_add': {
            'animation': 'Fade in, expand height from 0',
            'duration': '200ms',
            'easing': 'ease-out'
        },
        'modal_open': {
            'animation': 'Backdrop fades in, modal scales from 95% to 100%',
            'duration': '200ms',
            'easing': 'ease-out'
        }
    }
}
```

## Figma Workflow

### File Organization

```python
FIGMA_FILE_STRUCTURE = {
    'pages': {
        '🎨 Cover': 'Project thumbnail and metadata',
        '📋 Documentation': 'Design specs, decisions, changelog',
        '🔄 Flows': 'User flow diagrams',
        '📱 Mobile': 'Mobile screen designs',
        '💻 Desktop': 'Desktop screen designs',
        '🧩 Components': 'Component library (if not using separate file)',
        '🗄️ Archive': 'Old versions, explorations'
    },
    'naming_convention': {
        'frames': '[Platform] / [Section] / [Screen Name] / [State]',
        'example': 'Mobile / Onboarding / Welcome / Default',
        'components': '[Category] / [Component] / [Variant]',
        'comp_example': 'Forms / Input / Default'
    },
    'status_indicators': {
        '🟢 Ready for dev': 'Approved, specs complete',
        '🟡 In review': 'Awaiting feedback',
        '🔴 WIP': 'Work in progress',
        '⚪ Exploration': 'Not final'
    }
}

class FigmaFrameNaming:
    """Generate consistent frame names"""

    def __init__(self, platform: str, section: str):
        self.platform = platform
        self.section = section

    def screen(self, name: str, state: str = 'Default') -> str:
        return f"{self.platform} / {self.section} / {name} / {state}"

    def component(self, category: str, name: str, variant: str = 'Default') -> str:
        return f"{category} / {name} / {variant}"

# Usage
mobile = FigmaFrameNaming('Mobile', 'Checkout')
frame_name = mobile.screen('Payment', 'Error')
# "Mobile / Checkout / Payment / Error"
```

### Figma API Integration

```python
import requests
from typing import Dict, List, Optional

class FigmaAPI:
    """
    Figma REST API client for automation

    API docs: https://www.figma.com/developers/api
    """

    def __init__(self, access_token: str):
        self.base_url = "https://api.figma.com/v1"
        self.headers = {
            "X-Figma-Token": access_token
        }

    def get_file(self, file_key: str) -> Dict:
        """
        Get file metadata and structure

        file_key: From Figma URL figma.com/file/{file_key}/...
        """
        response = requests.get(
            f"{self.base_url}/files/{file_key}",
            headers=self.headers
        )
        return response.json()

    def get_file_nodes(self, file_key: str, node_ids: List[str]) -> Dict:
        """Get specific nodes from file"""
        ids = ",".join(node_ids)
        response = requests.get(
            f"{self.base_url}/files/{file_key}/nodes",
            headers=self.headers,
            params={"ids": ids}
        )
        return response.json()

    def get_images(
        self,
        file_key: str,
        node_ids: List[str],
        format: str = "png",
        scale: float = 2
    ) -> Dict:
        """
        Export frames/components as images

        format: png, jpg, svg, pdf
        scale: 1-4 for raster formats
        """
        ids = ",".join(node_ids)
        response = requests.get(
            f"{self.base_url}/images/{file_key}",
            headers=self.headers,
            params={
                "ids": ids,
                "format": format,
                "scale": scale
            }
        )
        return response.json()

    def get_comments(self, file_key: str) -> Dict:
        """Get all comments on file"""
        response = requests.get(
            f"{self.base_url}/files/{file_key}/comments",
            headers=self.headers
        )
        return response.json()

    def post_comment(self, file_key: str, message: str, node_id: Optional[str] = None) -> Dict:
        """Add comment to file or specific node"""
        data = {"message": message}
        if node_id:
            data["client_meta"] = {"node_id": node_id}

        response = requests.post(
            f"{self.base_url}/files/{file_key}/comments",
            headers=self.headers,
            json=data
        )
        return response.json()

    def get_components(self, file_key: str) -> Dict:
        """Get all components in file"""
        response = requests.get(
            f"{self.base_url}/files/{file_key}/components",
            headers=self.headers
        )
        return response.json()

    def get_styles(self, file_key: str) -> Dict:
        """Get all styles (colors, text, effects) in file"""
        response = requests.get(
            f"{self.base_url}/files/{file_key}/styles",
            headers=self.headers
        )
        return response.json()


# Example: Export all frames as PNGs
def export_screens(figma: FigmaAPI, file_key: str, output_dir: str):
    """Export all top-level frames as images"""
    import os

    file_data = figma.get_file(file_key)

    # Find all frame nodes
    frame_ids = []
    for page in file_data['document']['children']:
        for child in page.get('children', []):
            if child['type'] == 'FRAME':
                frame_ids.append(child['id'])

    # Export images
    if frame_ids:
        images = figma.get_images(file_key, frame_ids)
        for node_id, url in images.get('images', {}).items():
            if url:
                response = requests.get(url)
                filename = f"{node_id.replace(':', '_')}.png"
                with open(os.path.join(output_dir, filename), 'wb') as f:
                    f.write(response.content)
```

### Design Tokens from Figma

```python
def extract_tokens_from_figma(figma: FigmaAPI, file_key: str) -> Dict:
    """
    Extract design tokens from Figma styles

    Returns tokens in Style Dictionary format
    """
    styles = figma.get_styles(file_key)

    tokens = {
        'color': {},
        'typography': {},
        'effect': {}
    }

    for style in styles.get('meta', {}).get('styles', []):
        style_type = style['style_type']
        name = style['name'].lower().replace(' ', '-').replace('/', '-')

        if style_type == 'FILL':
            # Color tokens
            tokens['color'][name] = {
                'value': style.get('description', ''),  # Would need node data for actual value
                'description': style.get('description', '')
            }
        elif style_type == 'TEXT':
            # Typography tokens
            tokens['typography'][name] = {
                'value': {},  # Would need node data for actual values
                'description': style.get('description', '')
            }
        elif style_type == 'EFFECT':
            # Shadow/blur tokens
            tokens['effect'][name] = {
                'value': {},
                'description': style.get('description', '')
            }

    return tokens
```

## Responsive Design

### Breakpoint Strategy

```python
RESPONSIVE_STRATEGY = {
    'mobile_first': {
        'description': 'Design for mobile, enhance for larger screens',
        'breakpoints': {
            'base': '0px (mobile)',
            'sm': '640px (large phones)',
            'md': '768px (tablets)',
            'lg': '1024px (laptops)',
            'xl': '1280px (desktops)',
            '2xl': '1536px (large screens)'
        },
        'when_to_use': [
            "Mobile is primary use case",
            "Progressive enhancement approach",
            "Content-first design"
        ]
    },
    'desktop_first': {
        'description': 'Design for desktop, adapt for smaller screens',
        'when_to_use': [
            "Desktop is primary use case (B2B, dashboards)",
            "Complex layouts that simplify on mobile",
            "Existing desktop product"
        ]
    }
}

class ResponsiveWireframe:
    """Plan responsive layouts"""

    def __init__(self, content_blocks: List[str]):
        self.content = content_blocks

    def layout_for_breakpoint(self, breakpoint: str) -> Dict:
        """Define layout at each breakpoint"""
        layouts = {
            'mobile': {
                'columns': 1,
                'stack': 'vertical',
                'hidden': [],  # Elements to hide
                'collapsed': ['nav', 'sidebar']  # Elements to collapse
            },
            'tablet': {
                'columns': 2,
                'stack': 'mixed',
                'hidden': [],
                'collapsed': []
            },
            'desktop': {
                'columns': 3,
                'stack': 'horizontal',
                'hidden': [],
                'collapsed': []
            }
        }
        return layouts.get(breakpoint, layouts['desktop'])

    def content_priority(self) -> List[Dict]:
        """Define content priority for mobile"""
        return [
            {'content': 'primary_action', 'priority': 1, 'mobile_position': 'sticky_bottom'},
            {'content': 'main_content', 'priority': 2, 'mobile_position': 'default'},
            {'content': 'navigation', 'priority': 3, 'mobile_position': 'hamburger'},
            {'content': 'sidebar', 'priority': 4, 'mobile_position': 'collapsed'}
        ]
```

## Design Handoff

### Specification Documentation

```python
HANDOFF_CHECKLIST = {
    'visual_specs': [
        "All screens at each breakpoint",
        "Component states (default, hover, active, disabled, error)",
        "Loading states",
        "Empty states",
        "Error states"
    ],
    'interaction_specs': [
        "Clickable prototype covering main flows",
        "Transition/animation specifications",
        "Gesture interactions (mobile)",
        "Keyboard interactions"
    ],
    'technical_specs': [
        "Design tokens (colors, typography, spacing)",
        "Asset exports (icons, images)",
        "Grid and spacing documentation",
        "Responsive behavior notes"
    ],
    'content': [
        "Final copy for all screens",
        "Character limits for dynamic content",
        "Placeholder/fallback content",
        "Localization considerations"
    ],
    'accessibility': [
        "Color contrast verification",
        "Focus order documentation",
        "Alt text for images",
        "ARIA requirements"
    ]
}

class DesignSpec:
    """Generate design specifications for handoff"""

    def __init__(self, component_name: str):
        self.name = component_name
        self.specs = {}

    def add_dimensions(self, width: str, height: str, padding: Dict, margin: Dict):
        self.specs['dimensions'] = {
            'width': width,
            'height': height,
            'padding': padding,
            'margin': margin
        }

    def add_typography(self, font: str, size: str, weight: str, line_height: str, color: str):
        self.specs['typography'] = {
            'font_family': font,
            'font_size': size,
            'font_weight': weight,
            'line_height': line_height,
            'color': color
        }

    def add_states(self, states: Dict[str, Dict]):
        """Add visual states (default, hover, active, disabled)"""
        self.specs['states'] = states

    def to_markdown(self) -> str:
        """Export as markdown specification"""
        md = f"# {self.name} Specifications\n\n"

        if 'dimensions' in self.specs:
            md += "## Dimensions\n"
            for key, value in self.specs['dimensions'].items():
                md += f"- **{key}**: {value}\n"

        if 'typography' in self.specs:
            md += "\n## Typography\n"
            for key, value in self.specs['typography'].items():
                md += f"- **{key}**: {value}\n"

        if 'states' in self.specs:
            md += "\n## States\n"
            for state, props in self.specs['states'].items():
                md += f"\n### {state}\n"
                for key, value in props.items():
                    md += f"- {key}: {value}\n"

        return md
```

## Best Practices

1. **Start low, increase fidelity**: Don't jump to high-fidelity too early
2. **Test with prototypes**: Real user feedback > assumptions
3. **Document decisions**: Why, not just what
4. **Consistent naming**: Team can find things quickly
5. **Version control**: Track changes, enable rollback
6. **Accessibility from start**: Not an afterthought

## Common Pitfalls

- **Premature polish**: High-fidelity before validating structure
- **Static designs**: Missing interactions and states
- **Pixel perfection obsession**: Spending time on details that don't matter yet
- **No responsive planning**: Only designing for one viewport
- **Poor handoff**: Missing specs, states, edge cases
- **Not testing prototypes**: Building without user validation

---

**Skill Type**: UX - Wireframing & Prototyping
**Complexity**: Beginner to Intermediate
**Typical Usage**: Design exploration, user testing, developer handoff
