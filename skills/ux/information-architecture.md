---
name: information-architecture
description: Load when user needs site structure, navigation patterns, content organization, taxonomies, or card sorting analysis. Covers IA principles and validation methods.
trigger_keywords: [information architecture, ia, site map, navigation, taxonomy, content strategy, card sorting, tree testing, findability, wayfinding, labeling, mental model, site structure, breadcrumbs]
---

# Information Architecture Skill

Organizing and structuring content for optimal findability and user comprehension.

## IA Fundamentals

### The IA Triad

```python
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class IATriad:
    """
    Rosenfeld & Morville's IA Triad

    Effective IA balances three interdependent factors
    """

    users: Dict = None  # Who uses it
    content: Dict = None  # What is being organized
    context: Dict = None  # Business goals and constraints

    def __post_init__(self):
        self.users = {
            'needs': [],  # What users are trying to accomplish
            'behaviors': [],  # How they search/browse
            'mental_models': [],  # How they expect info to be organized
            'vocabulary': []  # Language they use
        }
        self.content = {
            'volume': '',  # How much content
            'types': [],  # Documents, products, articles, etc.
            'structure': '',  # Existing organization
            'ownership': [],  # Who creates/maintains
            'metadata': []  # Available attributes
        }
        self.context = {
            'business_goals': [],  # What org wants to achieve
            'constraints': [],  # Tech, budget, timeline
            'stakeholders': [],  # Decision makers
            'governance': ''  # How content is managed
        }


# IA Components
IA_COMPONENTS = {
    'organization_systems': {
        'description': 'How content is categorized and structured',
        'schemes': ['Topic', 'Task', 'Audience', 'Chronological', 'Alphabetical'],
        'structures': ['Hierarchy', 'Database', 'Hypertext', 'Linear']
    },
    'labeling_systems': {
        'description': 'How content is named/represented',
        'types': ['Navigation labels', 'Contextual links', 'Headings', 'Index terms']
    },
    'navigation_systems': {
        'description': 'How users move through content',
        'types': ['Global nav', 'Local nav', 'Contextual nav', 'Supplemental nav']
    },
    'search_systems': {
        'description': 'How users query for content',
        'components': ['Search interface', 'Query processing', 'Results display', 'Filters']
    }
}
```

### Organization Schemes

```python
class OrganizationScheme:
    """
    How to categorize content

    Exact schemes: Objectively organized
    Ambiguous schemes: Subjectively organized
    """

    EXACT_SCHEMES = {
        'alphabetical': {
            'use_when': "Large known-item sets (directories, glossaries)",
            'pros': "No interpretation needed",
            'cons': "Requires knowing exact name"
        },
        'chronological': {
            'use_when': "Time-sensitive content (news, archives, events)",
            'pros': "Shows recency, progression",
            'cons': "Topic findability poor"
        },
        'geographical': {
            'use_when': "Location-based services (stores, delivery)",
            'pros': "Natural for physical things",
            'cons': "Limited to location-relevant content"
        }
    }

    AMBIGUOUS_SCHEMES = {
        'topic': {
            'use_when': "Most common; subject-based organization",
            'example': "Products by category: Electronics > Computers > Laptops",
            'challenge': "Cross-category items; whose mental model?"
        },
        'task': {
            'use_when': "Action-oriented users",
            'example': "File taxes, Register vehicle, Pay bills",
            'challenge': "Tasks may span categories"
        },
        'audience': {
            'use_when': "Distinct user groups with different needs",
            'example': "For Students, For Faculty, For Parents",
            'challenge': "User may not self-identify; overlap"
        },
        'metaphor': {
            'use_when': "Explaining unfamiliar concepts",
            'example': "Desktop metaphor (files, folders, trash)",
            'challenge': "Can limit future growth; cultural issues"
        }
    }

    @staticmethod
    def recommend_scheme(
        content_type: str,
        user_behavior: str,
        content_volume: int
    ) -> str:
        """Recommend organization scheme based on context"""
        if user_behavior == 'known_item_search':
            return 'alphabetical'
        elif content_type == 'time_sensitive':
            return 'chronological'
        elif user_behavior == 'task_oriented':
            return 'task'
        elif content_volume > 1000:
            return 'topic'  # Faceted
        else:
            return 'topic'  # Simple hierarchy
```

## Site Structure

### Hierarchical Structures

```python
from typing import Optional
import json

@dataclass
class SiteNode:
    """Single node in site hierarchy"""

    id: str
    label: str
    url: str
    parent: Optional[str] = None
    children: List['SiteNode'] = None
    metadata: Dict = None

    def __post_init__(self):
        self.children = self.children or []
        self.metadata = self.metadata or {}

class SiteMap:
    """
    Site structure representation

    Depth guidelines:
    - 3 clicks rule (debunked but useful guideline)
    - Actually: Minimize cognitive load, not clicks
    - Wide vs Deep: Prefer wider, shallower structures
    """

    def __init__(self, name: str):
        self.name = name
        self.root = SiteNode(id='root', label='Home', url='/')
        self.nodes = {'root': self.root}

    def add_page(
        self,
        id: str,
        label: str,
        url: str,
        parent_id: str = 'root',
        metadata: Dict = None
    ) -> SiteNode:
        """Add page to site structure"""
        node = SiteNode(id=id, label=label, url=url, parent=parent_id, metadata=metadata)
        self.nodes[id] = node

        if parent_id in self.nodes:
            self.nodes[parent_id].children.append(node)

        return node

    def get_depth(self, node_id: str) -> int:
        """Calculate depth of node from root"""
        depth = 0
        current = self.nodes.get(node_id)
        while current and current.parent:
            depth += 1
            current = self.nodes.get(current.parent)
        return depth

    def get_breadcrumb(self, node_id: str) -> List[SiteNode]:
        """Generate breadcrumb trail"""
        breadcrumb = []
        current = self.nodes.get(node_id)
        while current:
            breadcrumb.insert(0, current)
            current = self.nodes.get(current.parent) if current.parent else None
        return breadcrumb

    def analyze_structure(self) -> Dict:
        """Analyze site structure health"""
        depths = [self.get_depth(nid) for nid in self.nodes]
        widths = [len(n.children) for n in self.nodes.values()]

        return {
            'total_pages': len(self.nodes),
            'max_depth': max(depths),
            'avg_depth': sum(depths) / len(depths),
            'max_width': max(widths),
            'avg_width': sum(widths) / len(widths),
            'orphans': [n.id for n in self.nodes.values() if not n.parent and n.id != 'root'],
            'deep_pages': [nid for nid in self.nodes if self.get_depth(nid) > 4],
            'wide_nodes': [n.id for n in self.nodes.values() if len(n.children) > 7]
        }

    def to_json(self) -> str:
        """Export as JSON for visualization tools"""
        def node_to_dict(node: SiteNode) -> Dict:
            return {
                'id': node.id,
                'label': node.label,
                'url': node.url,
                'children': [node_to_dict(c) for c in node.children]
            }
        return json.dumps(node_to_dict(self.root), indent=2)


# Example site map
ecommerce_site = SiteMap("E-commerce Store")
ecommerce_site.add_page('products', 'Products', '/products')
ecommerce_site.add_page('electronics', 'Electronics', '/products/electronics', 'products')
ecommerce_site.add_page('laptops', 'Laptops', '/products/electronics/laptops', 'electronics')
ecommerce_site.add_page('clothing', 'Clothing', '/products/clothing', 'products')
ecommerce_site.add_page('about', 'About Us', '/about')
ecommerce_site.add_page('support', 'Support', '/support')
ecommerce_site.add_page('faq', 'FAQ', '/support/faq', 'support')
```

### Flat vs Deep Structures

```python
STRUCTURE_PATTERNS = {
    'flat': {
        'characteristics': "Few levels, many items per level",
        'pros': [
            "Fewer clicks to destination",
            "Easier to scan all options",
            "Simple mental model"
        ],
        'cons': [
            "Can be overwhelming (> 7-9 items)",
            "Hard to show relationships",
            "Mobile navigation challenging"
        ],
        'best_for': "Small sites, simple content, experienced users"
    },
    'deep': {
        'characteristics': "Many levels, few items per level",
        'pros': [
            "Clear categorization",
            "Less overwhelming per level",
            "Shows content relationships"
        ],
        'cons': [
            "More clicks required",
            "Easy to get lost",
            "Harder to change mental model"
        ],
        'best_for': "Large sites, complex taxonomies"
    },
    'hybrid': {
        'characteristics': "Moderate depth (3-4 levels), moderate width (5-7 items)",
        'recommendation': "Most common; balance findability and comprehension"
    }
}

def recommend_structure(
    total_pages: int,
    user_expertise: str,
    content_relationships: str
) -> str:
    """Recommend structure approach"""
    if total_pages < 50 and user_expertise == 'high':
        return 'flat'
    elif content_relationships == 'hierarchical' and total_pages > 200:
        return 'deep'
    else:
        return 'hybrid'
```

## Navigation Systems

### Navigation Types

```python
@dataclass
class NavigationSystem:
    """
    Navigation design patterns

    Embedded: Built into content pages
    Supplemental: Additional navigation aids
    """

    EMBEDDED_NAV = {
        'global': {
            'description': "Present on all pages, shows top-level structure",
            'placement': "Header, top of page",
            'items': "5-7 max for horizontal",
            'examples': ["Main menu", "Header nav", "Mega menu"]
        },
        'local': {
            'description': "Shows siblings/children in current section",
            'placement': "Left sidebar, below header",
            'purpose': "Navigate within section",
            'examples': ["Sidebar nav", "Section submenu"]
        },
        'contextual': {
            'description': "Embedded links within content",
            'placement': "Within page content",
            'purpose': "Related content, cross-links",
            'examples': ["Related articles", "See also", "Inline links"]
        }
    }

    SUPPLEMENTAL_NAV = {
        'sitemap': {
            'description': "Complete site structure on one page",
            'purpose': "Overview, backup navigation, SEO"
        },
        'index': {
            'description': "A-Z listing of content",
            'purpose': "Known-item finding, comprehensive access"
        },
        'search': {
            'description': "Query-based navigation",
            'purpose': "When users know what they want",
            'best_practices': [
                "Prominent search box",
                "Auto-suggest",
                "Filters/facets",
                "Clear results display"
            ]
        },
        'breadcrumbs': {
            'description': "Shows path from home to current page",
            'purpose': "Orientation, easy backtracking",
            'format': "Home > Section > Subsection > Page"
        },
        'tags_facets': {
            'description': "Filter/refine content by attributes",
            'purpose': "Narrow large content sets"
        }
    }


class MegaMenu:
    """
    Mega menu design patterns

    Good for: Large sites with many categories
    Avoid: Simple sites, mobile-first
    """

    def design_mega_menu(
        self,
        categories: List[Dict],
        max_columns: int = 5
    ) -> Dict:
        """Design mega menu structure"""
        return {
            'layout': 'columnar',
            'columns': min(len(categories), max_columns),
            'guidelines': [
                "Group related items visually",
                "Use clear category headings",
                "Limit depth to 2 levels in menu",
                "Include visual cues (icons, images)",
                "Ensure keyboard navigability",
                "Don't auto-open on hover (accessibility)"
            ],
            'categories': categories
        }
```

### Mobile Navigation Patterns

```python
MOBILE_NAV_PATTERNS = {
    'hamburger': {
        'description': "Three-line icon, slides in menu",
        'pros': "Space efficient, familiar",
        'cons': "Hidden content, extra tap",
        'best_for': "Complex nav, secondary features"
    },
    'tab_bar': {
        'description': "Bottom tabs, always visible",
        'pros': "Thumb-friendly, visible options",
        'cons': "Limited to 5 items max",
        'best_for': "Primary app sections, iOS standard"
    },
    'bottom_nav': {
        'description': "Bottom bar with icons + labels",
        'pros': "Accessible, visible, Material Design standard",
        'cons': "Takes screen space",
        'best_for': "3-5 primary destinations"
    },
    'hub_and_spoke': {
        'description': "Central hub page, branch to sections",
        'pros': "Clear structure, focused experience",
        'cons': "Requires return to hub",
        'best_for': "Distinct section apps"
    },
    'progressive_disclosure': {
        'description': "Show more as user drills down",
        'pros': "Reduces overwhelm",
        'cons': "Can hide important content",
        'best_for': "Complex content hierarchies"
    }
}
```

## Labeling

### Labeling Best Practices

```python
class LabelingSystem:
    """
    Clear, consistent labeling for navigation and content
    """

    PRINCIPLES = [
        "Use user vocabulary, not internal jargon",
        "Be specific, not vague ('Resources' is too broad)",
        "Be consistent across site",
        "Keep labels short (1-3 words for nav)",
        "Front-load important words",
        "Test labels with users (card sorting, tree testing)"
    ]

    COMMON_MISTAKES = [
        {"bad": "Solutions", "why": "Too vague", "better": "Products" or specific category},
        {"bad": "Resources", "why": "Catch-all, meaningless", "better": "Blog, Docs, Guides"},
        {"bad": "Info", "why": "Everything is info", "better": "About Us, Help"},
        {"bad": "Stuff", "why": "Too informal/vague", "better": Specific category},
        {"bad": "Misc", "why": "Orphan content", "better": Reorganize content}
    ]

    def evaluate_label(self, label: str) -> Dict:
        """Evaluate label quality"""
        issues = []

        if len(label) > 20:
            issues.append("Too long for navigation")
        if label.lower() in ['resources', 'solutions', 'info', 'stuff', 'misc']:
            issues.append("Too vague - be more specific")
        if not label[0].isupper():
            issues.append("Consider title case for nav labels")

        return {
            'label': label,
            'issues': issues,
            'quality': 'good' if not issues else 'needs_work'
        }

    def generate_alternatives(self, vague_label: str, content_types: List[str]) -> List[str]:
        """Suggest specific alternatives for vague labels"""
        alternatives = []
        for content_type in content_types:
            if content_type == 'articles':
                alternatives.append('Blog')
            elif content_type == 'documentation':
                alternatives.append('Docs')
            elif content_type == 'tutorials':
                alternatives.append('Learn')
            elif content_type == 'api_docs':
                alternatives.append('API Reference')
        return alternatives
```

## Taxonomy Design

### Taxonomy Structure

```python
@dataclass
class Taxonomy:
    """
    Controlled vocabulary and classification system
    """

    name: str
    terms: List[Dict] = None

    def __post_init__(self):
        self.terms = self.terms or []

    def add_term(
        self,
        term: str,
        broader: Optional[str] = None,
        narrower: List[str] = None,
        related: List[str] = None,
        synonyms: List[str] = None,
        definition: str = ""
    ):
        """Add term with relationships"""
        self.terms.append({
            'term': term,
            'broader': broader,  # Parent term (BT)
            'narrower': narrower or [],  # Child terms (NT)
            'related': related or [],  # Associated terms (RT)
            'synonyms': synonyms or [],  # Use For (UF)
            'definition': definition
        })

    def get_hierarchy(self) -> Dict:
        """Build hierarchical structure"""
        # Find root terms (no broader term)
        roots = [t for t in self.terms if not t['broader']]
        return self._build_tree(roots)

    def _build_tree(self, terms: List[Dict]) -> List[Dict]:
        """Recursively build tree structure"""
        tree = []
        for term in terms:
            children = [t for t in self.terms if t['broader'] == term['term']]
            tree.append({
                'term': term['term'],
                'children': self._build_tree(children)
            })
        return tree


# Example: Product taxonomy
product_taxonomy = Taxonomy(name="Product Categories")
product_taxonomy.add_term("Electronics", definition="Electronic devices and accessories")
product_taxonomy.add_term("Computers", broader="Electronics")
product_taxonomy.add_term("Laptops", broader="Computers", synonyms=["Notebooks"])
product_taxonomy.add_term("Desktops", broader="Computers", related=["Monitors"])
product_taxonomy.add_term("Phones", broader="Electronics", synonyms=["Mobile Phones", "Cell Phones"])

class FacetedClassification:
    """
    Faceted classification: Multiple independent dimensions

    Better than single hierarchy for complex content
    """

    def __init__(self, name: str):
        self.name = name
        self.facets = {}

    def add_facet(self, facet_name: str, values: List[str]):
        """Add classification dimension"""
        self.facets[facet_name] = values

    def classify_item(self, item_id: str, facet_values: Dict[str, str]) -> Dict:
        """Classify item across facets"""
        return {
            'item_id': item_id,
            'classification': facet_values
        }


# Example: Recipe faceted classification
recipe_classification = FacetedClassification("Recipes")
recipe_classification.add_facet("Cuisine", ["Italian", "Mexican", "Asian", "American"])
recipe_classification.add_facet("Meal Type", ["Breakfast", "Lunch", "Dinner", "Snack"])
recipe_classification.add_facet("Diet", ["Vegetarian", "Vegan", "Gluten-Free", "Keto"])
recipe_classification.add_facet("Difficulty", ["Easy", "Medium", "Hard"])
recipe_classification.add_facet("Time", ["Under 30 min", "30-60 min", "Over 1 hour"])
```

## IA Validation

### Card Sorting Analysis

```python
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

class CardSortAnalysis:
    """
    Analyze card sorting results
    """

    def __init__(self, items: List[str]):
        self.items = items
        self.sorts = []  # List of {participant_id, groups: {group_name: [items]}}

    def add_sort(self, participant_id: str, groups: Dict[str, List[str]]):
        """Add participant's card sort"""
        self.sorts.append({
            'participant': participant_id,
            'groups': groups
        })

    def similarity_matrix(self) -> np.ndarray:
        """
        Calculate item-to-item similarity matrix

        Similarity = % of participants who grouped items together
        """
        n = len(self.items)
        matrix = np.zeros((n, n))

        for i, item_a in enumerate(self.items):
            for j, item_b in enumerate(self.items):
                if i == j:
                    matrix[i][j] = 1.0
                else:
                    grouped_together = 0
                    for sort in self.sorts:
                        for group_items in sort['groups'].values():
                            if item_a in group_items and item_b in group_items:
                                grouped_together += 1
                                break
                    matrix[i][j] = grouped_together / len(self.sorts)

        return matrix

    def cluster_items(self) -> Dict:
        """
        Hierarchical clustering of items

        Use dendrogram to visualize suggested groupings
        """
        sim_matrix = self.similarity_matrix()
        # Convert similarity to distance
        dist_matrix = 1 - sim_matrix
        # Ensure symmetry and zero diagonal
        np.fill_diagonal(dist_matrix, 0)

        # Hierarchical clustering
        condensed = squareform(dist_matrix)
        linkage_matrix = linkage(condensed, method='average')

        return {
            'linkage': linkage_matrix,
            'items': self.items,
            'method': 'Average linkage clustering'
        }

    def category_agreement(self) -> Dict[str, float]:
        """
        How much do participants agree on category names?
        """
        # Collect all category names used
        all_categories = {}
        for sort in self.sorts:
            for cat_name in sort['groups'].keys():
                normalized = cat_name.lower().strip()
                all_categories[normalized] = all_categories.get(normalized, 0) + 1

        # Calculate agreement
        total_categories = sum(all_categories.values())
        return {
            cat: count / len(self.sorts)
            for cat, count in all_categories.items()
        }

    def problem_cards(self) -> List[Dict]:
        """
        Identify cards that are sorted inconsistently
        """
        problems = []
        sim_matrix = self.similarity_matrix()

        for i, item in enumerate(self.items):
            # Low maximum similarity = not consistently grouped with anything
            max_sim = max(sim_matrix[i][j] for j in range(len(self.items)) if j != i)
            if max_sim < 0.5:  # Less than 50% agreement
                problems.append({
                    'item': item,
                    'max_agreement': max_sim,
                    'issue': "Not consistently categorized"
                })

        return problems
```

### Tree Testing

```python
class TreeTest:
    """
    Tree Testing: Validate IA without visual design

    Give users tasks, see if they find correct location
    """

    def __init__(self, tree: Dict):
        """
        tree: Hierarchical structure
        {
            'label': 'Home',
            'children': [
                {'label': 'Products', 'children': [...]},
                ...
            ]
        }
        """
        self.tree = tree
        self.tasks = []
        self.results = []

    def add_task(
        self,
        task_description: str,
        correct_path: List[str],  # ['Products', 'Electronics', 'Laptops']
        correct_answer: str  # Final destination label
    ):
        """Add test task"""
        self.tasks.append({
            'task': task_description,
            'correct_path': correct_path,
            'correct_answer': correct_answer
        })

    def record_result(
        self,
        participant_id: str,
        task_id: int,
        path_taken: List[str],
        final_answer: str,
        time_seconds: float,
        confidence: int  # 1-5 scale
    ):
        """Record participant's answer"""
        task = self.tasks[task_id]
        self.results.append({
            'participant': participant_id,
            'task_id': task_id,
            'path_taken': path_taken,
            'final_answer': final_answer,
            'time': time_seconds,
            'confidence': confidence,
            'success': final_answer == task['correct_answer'],
            'direct_success': path_taken == task['correct_path']
        })

    def analyze_task(self, task_id: int) -> Dict:
        """Analyze results for specific task"""
        task_results = [r for r in self.results if r['task_id'] == task_id]

        if not task_results:
            return {'error': 'No results for task'}

        successes = [r for r in task_results if r['success']]
        direct = [r for r in task_results if r['direct_success']]

        return {
            'task': self.tasks[task_id]['task'],
            'correct_path': self.tasks[task_id]['correct_path'],
            'total_attempts': len(task_results),
            'success_rate': len(successes) / len(task_results),
            'directness_rate': len(direct) / len(task_results),
            'avg_time': sum(r['time'] for r in task_results) / len(task_results),
            'avg_confidence': sum(r['confidence'] for r in task_results) / len(task_results),
            'common_wrong_paths': self._common_wrong_paths(task_results)
        }

    def _common_wrong_paths(self, results: List[Dict]) -> List[Dict]:
        """Find common wrong paths taken"""
        wrong_paths = [
            tuple(r['path_taken'])
            for r in results
            if not r['success']
        ]
        path_counts = {}
        for path in wrong_paths:
            path_counts[path] = path_counts.get(path, 0) + 1

        return [
            {'path': list(path), 'count': count}
            for path, count in sorted(path_counts.items(), key=lambda x: -x[1])[:5]
        ]

    def overall_metrics(self) -> Dict:
        """Calculate overall tree test performance"""
        return {
            'overall_success_rate': sum(r['success'] for r in self.results) / len(self.results),
            'overall_directness': sum(r['direct_success'] for r in self.results) / len(self.results),
            'problem_tasks': [
                self.tasks[i]['task']
                for i in range(len(self.tasks))
                if self.analyze_task(i)['success_rate'] < 0.6
            ]
        }
```

## Best Practices

1. **Start with user research**: Understand mental models before structuring
2. **Use multiple organization schemes**: Support different ways of finding
3. **Validate early**: Card sort and tree test before building
4. **Keep it shallow**: 3-4 levels max; wider is better than deeper
5. **Label clearly**: Use user language, be specific
6. **Plan for growth**: Structure should accommodate future content

## Common Pitfalls

- **Org chart = site structure**: Internal structure ≠ user mental model
- **Too many choices**: 7±2 items per level; more causes paralysis
- **Vague labels**: "Resources", "Solutions", "Info" are meaningless
- **No search strategy**: Search is navigation; design it deliberately
- **Ignoring cross-links**: Content doesn't fit in one place; allow multiple paths
- **Static IA**: User needs and content change; IA should evolve

---

**Skill Type**: UX - Information Architecture
**Complexity**: Intermediate to Advanced
**Typical Usage**: Site redesigns, new product development, content strategy
