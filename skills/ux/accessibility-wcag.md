---
name: accessibility-wcag
description: Load when user needs WCAG compliance, accessibility testing, ARIA patterns, screen reader support, keyboard navigation, or inclusive design. Covers A, AA, AAA conformance levels.
trigger_keywords: [accessibility, wcag, a11y, aria, screen reader, keyboard navigation, color contrast, alt text, focus management, semantic html, assistive technology, inclusive design, ada compliance, section 508]
---

# Accessibility & WCAG Skill

Implementing accessible digital experiences following WCAG 2.1/2.2 guidelines.

## WCAG Overview

### WCAG Principles (POUR)

```python
from dataclasses import dataclass
from typing import List, Dict
from enum import Enum

class ConformanceLevel(Enum):
    A = "A"  # Minimum accessibility
    AA = "AA"  # Standard target (legal requirement in many jurisdictions)
    AAA = "AAA"  # Enhanced accessibility

@dataclass
class WCAGPrinciple:
    """WCAG is organized around four principles"""

    name: str
    description: str
    guidelines: List[str]

WCAG_PRINCIPLES = {
    'perceivable': WCAGPrinciple(
        name="Perceivable",
        description="Information must be presentable in ways users can perceive",
        guidelines=[
            "1.1 Text Alternatives",
            "1.2 Time-based Media",
            "1.3 Adaptable",
            "1.4 Distinguishable"
        ]
    ),
    'operable': WCAGPrinciple(
        name="Operable",
        description="User interface components must be operable",
        guidelines=[
            "2.1 Keyboard Accessible",
            "2.2 Enough Time",
            "2.3 Seizures and Physical Reactions",
            "2.4 Navigable",
            "2.5 Input Modalities"
        ]
    ),
    'understandable': WCAGPrinciple(
        name="Understandable",
        description="Information and UI operation must be understandable",
        guidelines=[
            "3.1 Readable",
            "3.2 Predictable",
            "3.3 Input Assistance"
        ]
    ),
    'robust': WCAGPrinciple(
        name="Robust",
        description="Content must be robust enough for assistive technologies",
        guidelines=[
            "4.1 Compatible"
        ]
    )
}
```

### Key Success Criteria

```python
CRITICAL_CRITERIA = {
    # Level A (must have)
    '1.1.1': {
        'name': "Non-text Content",
        'level': ConformanceLevel.A,
        'requirement': "All non-text content has text alternative",
        'techniques': [
            "Alt text for images",
            "Labels for form controls",
            "Text alternatives for media"
        ]
    },
    '1.3.1': {
        'name': "Info and Relationships",
        'level': ConformanceLevel.A,
        'requirement': "Information and relationships are programmatically determinable",
        'techniques': [
            "Semantic HTML (headings, lists, tables)",
            "ARIA landmarks",
            "Form labels associated with inputs"
        ]
    },
    '2.1.1': {
        'name': "Keyboard",
        'level': ConformanceLevel.A,
        'requirement': "All functionality available via keyboard",
        'techniques': [
            "No keyboard traps",
            "Logical tab order",
            "Visible focus indicators"
        ]
    },
    '2.4.4': {
        'name': "Link Purpose (In Context)",
        'level': ConformanceLevel.A,
        'requirement': "Link purpose determinable from link text or context",
        'techniques': [
            "Descriptive link text (not 'click here')",
            "ARIA labels when needed"
        ]
    },

    # Level AA (target for compliance)
    '1.4.3': {
        'name': "Contrast (Minimum)",
        'level': ConformanceLevel.AA,
        'requirement': "Text has 4.5:1 contrast ratio (3:1 for large text)",
        'techniques': [
            "Check all text colors",
            "Include focus states",
            "Test with contrast checker tools"
        ]
    },
    '1.4.4': {
        'name': "Resize Text",
        'level': ConformanceLevel.AA,
        'requirement': "Text can be resized to 200% without loss of content",
        'techniques': [
            "Use relative units (rem, em, %)",
            "Test at browser zoom levels",
            "Avoid fixed heights on text containers"
        ]
    },
    '2.4.7': {
        'name': "Focus Visible",
        'level': ConformanceLevel.AA,
        'requirement': "Keyboard focus indicator is visible",
        'techniques': [
            "Don't remove :focus outlines",
            "Custom focus styles must be visible",
            "3:1 contrast for focus indicators (WCAG 2.2)"
        ]
    }
}
```

## Semantic HTML

### Proper Structure

```html
<!-- Good: Semantic structure -->
<header>
  <nav aria-label="Main navigation">
    <ul>
      <li><a href="/">Home</a></li>
      <li><a href="/products">Products</a></li>
    </ul>
  </nav>
</header>

<main>
  <article>
    <h1>Page Title</h1>
    <p>Introduction paragraph...</p>

    <section>
      <h2>Section Heading</h2>
      <p>Section content...</p>
    </section>
  </article>

  <aside aria-label="Related content">
    <h2>Related Articles</h2>
    <ul>
      <li><a href="/article-1">Article 1</a></li>
    </ul>
  </aside>
</main>

<footer>
  <p>&copy; 2024 Company Name</p>
</footer>

<!-- Bad: Div soup -->
<div class="header">
  <div class="nav">
    <div class="nav-item">Home</div>
  </div>
</div>
<div class="main">
  <div class="title">Page Title</div>
</div>
```

### Heading Hierarchy

```python
def validate_heading_hierarchy(headings: List[str]) -> Dict:
    """
    Validate heading levels follow logical order

    Rules:
    - Start with h1 (only one per page)
    - Don't skip levels (h1 → h3 is invalid)
    - Levels can decrease by any amount
    """
    issues = []
    h1_count = 0
    prev_level = 0

    for heading in headings:
        level = int(heading[1])  # Extract number from 'h1', 'h2', etc.

        if level == 1:
            h1_count += 1
            if h1_count > 1:
                issues.append("Multiple h1 elements (should have only one)")

        if prev_level > 0 and level > prev_level + 1:
            issues.append(f"Skipped heading level: h{prev_level} → h{level}")

        prev_level = level

    return {
        'valid': len(issues) == 0,
        'issues': issues,
        'h1_count': h1_count
    }

# Example
headings = ['h1', 'h2', 'h3', 'h2', 'h4']  # h2 → h4 skips h3
result = validate_heading_hierarchy(headings)
# {'valid': False, 'issues': ['Skipped heading level: h2 → h4'], 'h1_count': 1}
```

## ARIA Patterns

### ARIA Roles and Attributes

```python
ARIA_LANDMARKS = {
    'banner': "Site header (use <header> instead if possible)",
    'navigation': "Navigation region (<nav>)",
    'main': "Main content (<main>)",
    'complementary': "Supporting content (<aside>)",
    'contentinfo': "Footer information (<footer>)",
    'search': "Search functionality",
    'form': "Form region (<form>)",
    'region': "Generic landmark (requires aria-label)"
}

ARIA_STATES = {
    'aria-expanded': {
        'usage': "Expandable widgets (accordions, dropdowns)",
        'values': ['true', 'false'],
        'example': '<button aria-expanded="false" aria-controls="panel1">Toggle</button>'
    },
    'aria-selected': {
        'usage': "Selected state in tabs, listbox",
        'values': ['true', 'false'],
        'example': '<div role="tab" aria-selected="true">Tab 1</div>'
    },
    'aria-hidden': {
        'usage': "Hide from assistive technology",
        'values': ['true', 'false'],
        'warning': "Content is completely hidden from screen readers"
    },
    'aria-live': {
        'usage': "Announce dynamic content changes",
        'values': ['polite', 'assertive', 'off'],
        'example': '<div aria-live="polite">Status updates here</div>'
    },
    'aria-disabled': {
        'usage': "Indicate disabled state",
        'values': ['true', 'false'],
        'note': "Doesn't prevent interaction; use with disabled attribute or JS"
    }
}

ARIA_PROPERTIES = {
    'aria-label': {
        'usage': "Provide accessible name when text isn't visible",
        'example': '<button aria-label="Close dialog">×</button>'
    },
    'aria-labelledby': {
        'usage': "Reference visible text as label",
        'example': '<div aria-labelledby="dialog-title" role="dialog">...'
    },
    'aria-describedby': {
        'usage': "Reference additional descriptive text",
        'example': '<input aria-describedby="password-requirements">'
    },
    'aria-controls': {
        'usage': "Identify element being controlled",
        'example': '<button aria-controls="menu1" aria-expanded="false">'
    }
}
```

### Common Widget Patterns

```html
<!-- Accessible Dropdown Menu -->
<div class="dropdown">
  <button
    id="menu-button"
    aria-haspopup="true"
    aria-expanded="false"
    aria-controls="menu-list"
  >
    Menu
  </button>
  <ul
    id="menu-list"
    role="menu"
    aria-labelledby="menu-button"
    hidden
  >
    <li role="menuitem"><a href="/option1">Option 1</a></li>
    <li role="menuitem"><a href="/option2">Option 2</a></li>
  </ul>
</div>

<script>
// Toggle menu
const button = document.getElementById('menu-button');
const menu = document.getElementById('menu-list');

button.addEventListener('click', () => {
  const expanded = button.getAttribute('aria-expanded') === 'true';
  button.setAttribute('aria-expanded', !expanded);
  menu.hidden = expanded;
});

// Keyboard navigation
menu.addEventListener('keydown', (e) => {
  const items = menu.querySelectorAll('[role="menuitem"]');
  const currentIndex = Array.from(items).indexOf(document.activeElement);

  switch (e.key) {
    case 'ArrowDown':
      e.preventDefault();
      items[(currentIndex + 1) % items.length].focus();
      break;
    case 'ArrowUp':
      e.preventDefault();
      items[(currentIndex - 1 + items.length) % items.length].focus();
      break;
    case 'Escape':
      button.setAttribute('aria-expanded', 'false');
      menu.hidden = true;
      button.focus();
      break;
  }
});
</script>

<!-- Accessible Tabs -->
<div class="tabs">
  <div role="tablist" aria-label="Sample Tabs">
    <button
      role="tab"
      aria-selected="true"
      aria-controls="panel-1"
      id="tab-1"
    >
      Tab 1
    </button>
    <button
      role="tab"
      aria-selected="false"
      aria-controls="panel-2"
      id="tab-2"
      tabindex="-1"
    >
      Tab 2
    </button>
  </div>

  <div
    role="tabpanel"
    id="panel-1"
    aria-labelledby="tab-1"
  >
    Panel 1 content
  </div>

  <div
    role="tabpanel"
    id="panel-2"
    aria-labelledby="tab-2"
    hidden
  >
    Panel 2 content
  </div>
</div>

<!-- Accessible Modal Dialog -->
<div
  role="dialog"
  aria-modal="true"
  aria-labelledby="dialog-title"
  aria-describedby="dialog-desc"
>
  <h2 id="dialog-title">Confirm Action</h2>
  <p id="dialog-desc">Are you sure you want to proceed?</p>
  <button>Cancel</button>
  <button>Confirm</button>
</div>
```

## Color & Contrast

### Contrast Requirements

```python
import math
from typing import Tuple

def relative_luminance(r: int, g: int, b: int) -> float:
    """
    Calculate relative luminance per WCAG formula

    RGB values should be 0-255
    """
    def adjust(c):
        c = c / 255
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

    return 0.2126 * adjust(r) + 0.7152 * adjust(g) + 0.0722 * adjust(b)

def contrast_ratio(color1: Tuple[int, int, int], color2: Tuple[int, int, int]) -> float:
    """
    Calculate contrast ratio between two colors

    WCAG requirements:
    - Normal text: 4.5:1 (AA), 7:1 (AAA)
    - Large text (18pt+ or 14pt bold): 3:1 (AA), 4.5:1 (AAA)
    - UI components: 3:1 (AA)
    """
    l1 = relative_luminance(*color1)
    l2 = relative_luminance(*color2)

    lighter = max(l1, l2)
    darker = min(l1, l2)

    return (lighter + 0.05) / (darker + 0.05)

def check_contrast(
    foreground: Tuple[int, int, int],
    background: Tuple[int, int, int],
    text_size: str = 'normal'
) -> Dict:
    """
    Check if color combination meets WCAG requirements
    """
    ratio = contrast_ratio(foreground, background)

    requirements = {
        'normal': {'AA': 4.5, 'AAA': 7.0},
        'large': {'AA': 3.0, 'AAA': 4.5},
        'ui': {'AA': 3.0, 'AAA': 3.0}
    }

    req = requirements.get(text_size, requirements['normal'])

    return {
        'ratio': round(ratio, 2),
        'passes_AA': ratio >= req['AA'],
        'passes_AAA': ratio >= req['AAA'],
        'required_AA': req['AA'],
        'required_AAA': req['AAA']
    }

# Example usage
white = (255, 255, 255)
dark_gray = (51, 51, 51)
light_gray = (170, 170, 170)

print(check_contrast(dark_gray, white))
# {'ratio': 12.63, 'passes_AA': True, 'passes_AAA': True}

print(check_contrast(light_gray, white))
# {'ratio': 2.23, 'passes_AA': False, 'passes_AAA': False}
```

### Color Blindness Considerations

```python
COLOR_BLINDNESS_TYPES = {
    'protanopia': {
        'affects': "Red perception (~1% of males)",
        'avoid': "Red/green distinctions without other cues"
    },
    'deuteranopia': {
        'affects': "Green perception (~1% of males)",
        'avoid': "Red/green distinctions without other cues"
    },
    'tritanopia': {
        'affects': "Blue perception (rare)",
        'avoid': "Blue/yellow distinctions without other cues"
    }
}

SAFE_COLOR_PRACTICES = [
    "Don't rely on color alone to convey information",
    "Use patterns, icons, or text labels alongside color",
    "Ensure sufficient contrast between adjacent colors",
    "Test with color blindness simulation tools",
    "Provide high-contrast mode option"
]

# CSS example: Using patterns alongside color
CSS_PATTERNS = """
/* Don't rely on color alone for status */
.status-success {
  color: #28a745;
  background: url('checkmark-icon.svg') no-repeat left center;
  padding-left: 24px;
}

.status-error {
  color: #dc3545;
  background: url('x-icon.svg') no-repeat left center;
  padding-left: 24px;
}

/* Chart with patterns, not just colors */
.chart-bar-1 { background: #0066cc; }
.chart-bar-2 { background: repeating-linear-gradient(
  45deg, #cc6600, #cc6600 2px, transparent 2px, transparent 4px
); }
.chart-bar-3 { background: repeating-linear-gradient(
  -45deg, #009933, #009933 2px, transparent 2px, transparent 4px
); }
"""
```

## Keyboard Navigation

### Focus Management

```javascript
// Focus trap for modals
class FocusTrap {
  constructor(container) {
    this.container = container;
    this.focusableElements = container.querySelectorAll(
      'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
    );
    this.firstElement = this.focusableElements[0];
    this.lastElement = this.focusableElements[this.focusableElements.length - 1];
  }

  activate() {
    this.container.addEventListener('keydown', this.handleKeyDown.bind(this));
    this.firstElement.focus();
  }

  deactivate() {
    this.container.removeEventListener('keydown', this.handleKeyDown.bind(this));
  }

  handleKeyDown(e) {
    if (e.key !== 'Tab') return;

    if (e.shiftKey) {
      // Shift + Tab
      if (document.activeElement === this.firstElement) {
        e.preventDefault();
        this.lastElement.focus();
      }
    } else {
      // Tab
      if (document.activeElement === this.lastElement) {
        e.preventDefault();
        this.firstElement.focus();
      }
    }
  }
}

// Skip link implementation
document.querySelector('.skip-link').addEventListener('click', (e) => {
  e.preventDefault();
  const main = document.getElementById('main-content');
  main.tabIndex = -1;
  main.focus();
});

// Roving tabindex for composite widgets
class RovingTabindex {
  constructor(container, itemSelector) {
    this.container = container;
    this.items = container.querySelectorAll(itemSelector);
    this.currentIndex = 0;

    this.init();
  }

  init() {
    // Set initial tabindex
    this.items.forEach((item, index) => {
      item.tabIndex = index === 0 ? 0 : -1;
    });

    this.container.addEventListener('keydown', this.handleKeyDown.bind(this));
  }

  handleKeyDown(e) {
    let newIndex = this.currentIndex;

    switch (e.key) {
      case 'ArrowRight':
      case 'ArrowDown':
        newIndex = (this.currentIndex + 1) % this.items.length;
        break;
      case 'ArrowLeft':
      case 'ArrowUp':
        newIndex = (this.currentIndex - 1 + this.items.length) % this.items.length;
        break;
      case 'Home':
        newIndex = 0;
        break;
      case 'End':
        newIndex = this.items.length - 1;
        break;
      default:
        return;
    }

    e.preventDefault();
    this.items[this.currentIndex].tabIndex = -1;
    this.items[newIndex].tabIndex = 0;
    this.items[newIndex].focus();
    this.currentIndex = newIndex;
  }
}
```

### Visible Focus Styles

```css
/* Don't remove focus outlines - enhance them */
:focus {
  outline: 2px solid #005fcc;
  outline-offset: 2px;
}

/* Use :focus-visible for keyboard-only focus */
:focus:not(:focus-visible) {
  outline: none;
}

:focus-visible {
  outline: 2px solid #005fcc;
  outline-offset: 2px;
}

/* High contrast focus for dark backgrounds */
.dark-theme :focus-visible {
  outline: 2px solid #ffffff;
  box-shadow: 0 0 0 4px #005fcc;
}

/* Skip link - visible only on focus */
.skip-link {
  position: absolute;
  top: -40px;
  left: 0;
  background: #000;
  color: #fff;
  padding: 8px;
  z-index: 100;
}

.skip-link:focus {
  top: 0;
}
```

## Forms & Inputs

### Accessible Form Patterns

```html
<!-- Proper label association -->
<div class="form-group">
  <label for="email">Email address</label>
  <input
    type="email"
    id="email"
    name="email"
    aria-describedby="email-hint email-error"
    aria-invalid="false"
    required
  >
  <p id="email-hint" class="hint">We'll never share your email.</p>
  <p id="email-error" class="error" hidden>Please enter a valid email.</p>
</div>

<!-- Required field indication -->
<label for="name">
  Name <span aria-hidden="true">*</span>
  <span class="visually-hidden">(required)</span>
</label>
<input type="text" id="name" required aria-required="true">

<!-- Error message pattern -->
<div class="form-group" aria-live="polite">
  <label for="password">Password</label>
  <input
    type="password"
    id="password"
    aria-invalid="true"
    aria-describedby="password-error"
  >
  <p id="password-error" class="error" role="alert">
    Password must be at least 8 characters.
  </p>
</div>

<!-- Fieldset for related inputs -->
<fieldset>
  <legend>Shipping Address</legend>

  <div class="form-group">
    <label for="street">Street</label>
    <input type="text" id="street" autocomplete="street-address">
  </div>

  <div class="form-group">
    <label for="city">City</label>
    <input type="text" id="city" autocomplete="address-level2">
  </div>
</fieldset>

<!-- Radio group -->
<fieldset>
  <legend>Preferred contact method</legend>

  <div>
    <input type="radio" id="contact-email" name="contact" value="email">
    <label for="contact-email">Email</label>
  </div>

  <div>
    <input type="radio" id="contact-phone" name="contact" value="phone">
    <label for="contact-phone">Phone</label>
  </div>
</fieldset>
```

### Form Validation

```javascript
class AccessibleFormValidation {
  constructor(form) {
    this.form = form;
    this.form.addEventListener('submit', this.handleSubmit.bind(this));
  }

  handleSubmit(e) {
    const errors = this.validate();

    if (errors.length > 0) {
      e.preventDefault();
      this.announceErrors(errors);
      this.focusFirstError(errors);
    }
  }

  validate() {
    const errors = [];
    const requiredFields = this.form.querySelectorAll('[required]');

    requiredFields.forEach(field => {
      if (!field.value.trim()) {
        errors.push({
          field: field,
          message: `${field.labels[0].textContent} is required`
        });
        field.setAttribute('aria-invalid', 'true');
      }
    });

    return errors;
  }

  announceErrors(errors) {
    // Create or update live region
    let liveRegion = document.getElementById('form-errors');
    if (!liveRegion) {
      liveRegion = document.createElement('div');
      liveRegion.id = 'form-errors';
      liveRegion.setAttribute('role', 'alert');
      liveRegion.setAttribute('aria-live', 'assertive');
      this.form.prepend(liveRegion);
    }

    liveRegion.innerHTML = `
      <p>${errors.length} error(s) found:</p>
      <ul>
        ${errors.map(e => `<li>${e.message}</li>`).join('')}
      </ul>
    `;
  }

  focusFirstError(errors) {
    if (errors.length > 0) {
      errors[0].field.focus();
    }
  }
}
```

## Testing & Tools

### Automated Testing

```python
AUTOMATED_TOOLS = {
    'axe': {
        'type': 'Browser extension + API',
        'coverage': "~30-40% of issues",
        'url': "https://www.deque.com/axe/",
        'integration': ['Jest', 'Cypress', 'Playwright']
    },
    'lighthouse': {
        'type': 'Chrome DevTools',
        'coverage': "Basic accessibility audit",
        'url': "Built into Chrome"
    },
    'wave': {
        'type': 'Browser extension',
        'coverage': "Visual overlay of issues",
        'url': "https://wave.webaim.org/"
    },
    'pa11y': {
        'type': 'CLI tool',
        'coverage': "CI/CD integration",
        'url': "https://pa11y.org/"
    }
}

# Jest + axe example
JEST_AXE_EXAMPLE = """
import { axe, toHaveNoViolations } from 'jest-axe';

expect.extend(toHaveNoViolations);

test('component is accessible', async () => {
  const { container } = render(<MyComponent />);
  const results = await axe(container);
  expect(results).toHaveNoViolations();
});
"""

# Playwright accessibility test
PLAYWRIGHT_EXAMPLE = """
import { test, expect } from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';

test('page is accessible', async ({ page }) => {
  await page.goto('/');

  const results = await new AxeBuilder({ page }).analyze();

  expect(results.violations).toEqual([]);
});
"""
```

### Manual Testing Checklist

```python
MANUAL_TEST_CHECKLIST = {
    'keyboard': [
        "Tab through entire page - all interactive elements reachable?",
        "Visible focus indicator on all focused elements?",
        "Can operate all controls with keyboard?",
        "No keyboard traps?",
        "Skip link works?"
    ],
    'screen_reader': [
        "All content announced in logical order?",
        "Images have meaningful alt text?",
        "Form fields have labels announced?",
        "Errors announced when they occur?",
        "Dynamic content updates announced?"
    ],
    'zoom': [
        "Content readable at 200% zoom?",
        "No horizontal scrolling at 320px width?",
        "Text doesn't overlap or get cut off?"
    ],
    'color': [
        "Content understandable without color?",
        "Contrast ratios meet requirements?",
        "Focus indicators visible?"
    ]
}

SCREEN_READERS = {
    'NVDA': {'platform': 'Windows', 'cost': 'Free', 'browser': 'Firefox'},
    'JAWS': {'platform': 'Windows', 'cost': 'Paid', 'browser': 'Chrome/IE'},
    'VoiceOver': {'platform': 'macOS/iOS', 'cost': 'Built-in', 'browser': 'Safari'},
    'TalkBack': {'platform': 'Android', 'cost': 'Built-in', 'browser': 'Chrome'}
}
```

## Best Practices

1. **Use semantic HTML first**: ARIA is a supplement, not replacement
2. **Test with real users**: Automated tools catch <50% of issues
3. **Design accessibly from start**: Retrofitting is expensive
4. **Document accessibility decisions**: Include in design system
5. **Train the team**: Everyone's responsibility, not just developers

## Common Pitfalls

- **Div soup**: Using divs for everything instead of semantic elements
- **Missing alt text**: Or using "image" as alt text
- **Removing focus outlines**: Without providing alternative
- **Relying on color alone**: For errors, status, information
- **Keyboard traps**: Modal dialogs, date pickers, custom widgets
- **Auto-playing media**: Without user control or captions
- **ARIA overuse**: More ARIA ≠ more accessible

---

**Skill Type**: UX - Accessibility
**Complexity**: Intermediate to Advanced
**Typical Usage**: Development, QA, compliance audits
