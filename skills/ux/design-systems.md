---
name: design-systems
description: Load when user needs component libraries, design tokens, documentation patterns, Storybook setup, or design system governance. Covers atomic design and multi-platform consistency.
trigger_keywords: [design system, component library, design tokens, storybook, atomic design, ui kit, style guide, pattern library, brand guidelines, theme, variants, component api]
---

# Design Systems Skill

Building and maintaining scalable design systems for consistent user interfaces.

## Core Concepts

- **Design Tokens**: Named, platform-agnostic values (colors, spacing, typography) that create a shared language between design and code, enabling theme switching and multi-platform consistency
- **Atomic Design Hierarchy**: Components structured as Atoms (buttons, inputs) to Molecules (form fields) to Organisms (headers) to Templates to Pages - ensuring composability and reuse
- **Component API Design**: Well-designed props with sensible defaults, variant support, and extensibility points (className, style) that balance flexibility with consistency
- **Single Source of Truth**: One canonical definition for each component/token, with automated build pipelines exporting to CSS, JS, iOS, Android from shared source
- **Governance Model**: Contribution workflows, versioning strategy (semver), deprecation policies, and ownership structure that enable sustainable system evolution

## Design System Foundations

### What's in a Design System

```python
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum

@dataclass
class DesignSystem:
    """
    Complete design system structure

    A design system is more than a component library:
    - Design principles
    - Design tokens
    - Components
    - Patterns
    - Guidelines
    - Documentation
    """

    name: str

    # Core elements
    principles: List[str] = field(default_factory=list)
    tokens: Dict = field(default_factory=dict)
    components: List['Component'] = field(default_factory=list)
    patterns: List['Pattern'] = field(default_factory=list)

    # Documentation
    guidelines: Dict[str, str] = field(default_factory=dict)
    examples: List[Dict] = field(default_factory=list)

    # Governance
    owners: List[str] = field(default_factory=list)
    contribution_guide: str = ""
    versioning: str = "semver"


DESIGN_SYSTEM_LAYERS = {
    'foundations': {
        'description': 'Core design decisions',
        'includes': ['Colors', 'Typography', 'Spacing', 'Grid', 'Iconography', 'Motion']
    },
    'tokens': {
        'description': 'Named design values',
        'includes': ['Color tokens', 'Spacing scale', 'Font stacks', 'Shadow values']
    },
    'components': {
        'description': 'Reusable UI building blocks',
        'includes': ['Buttons', 'Inputs', 'Cards', 'Modals', 'Navigation']
    },
    'patterns': {
        'description': 'Common interaction patterns',
        'includes': ['Forms', 'Authentication', 'Search', 'Data tables', 'Onboarding']
    },
    'templates': {
        'description': 'Page-level layouts',
        'includes': ['Marketing pages', 'Dashboard', 'Settings', 'Empty states']
    }
}
```

### Atomic Design

```python
class AtomicDesign:
    """
    Brad Frost's Atomic Design methodology

    Atoms → Molecules → Organisms → Templates → Pages
    """

    LEVELS = {
        'atoms': {
            'description': "Basic building blocks (can't be broken down further)",
            'examples': ['Button', 'Input', 'Label', 'Icon', 'Avatar'],
            'characteristics': [
                "Single responsibility",
                "Highly reusable",
                "No business logic"
            ]
        },
        'molecules': {
            'description': "Groups of atoms working together",
            'examples': ['Search field (input + button)', 'Form field (label + input + error)'],
            'characteristics': [
                "Combines 2-3 atoms",
                "Still fairly abstract",
                "Focused purpose"
            ]
        },
        'organisms': {
            'description': "Complex UI sections",
            'examples': ['Header', 'Card', 'Comment thread', 'Product listing'],
            'characteristics': [
                "Multiple molecules/atoms",
                "Distinct section of interface",
                "May have state/behavior"
            ]
        },
        'templates': {
            'description': "Page-level layout structure",
            'examples': ['Article template', 'Dashboard layout', 'Checkout flow'],
            'characteristics': [
                "Arranges organisms",
                "Shows content structure",
                "Placeholder content"
            ]
        },
        'pages': {
            'description': "Specific instances with real content",
            'examples': ['Home page', 'Product detail page', 'User profile'],
            'characteristics': [
                "Real content",
                "Highest fidelity",
                "Test edge cases"
            ]
        }
    }

    @staticmethod
    def classify_component(name: str, dependencies: List[str]) -> str:
        """Determine atomic level based on dependencies"""
        if not dependencies:
            return 'atom'
        elif len(dependencies) <= 3 and all(d in ['atom'] for d in dependencies):
            return 'molecule'
        else:
            return 'organism'
```

## Design Tokens

### Token Structure

```python
from typing import Union

@dataclass
class DesignToken:
    """
    Design token: Named, platform-agnostic design value

    Tokens create a shared language between design and code
    """

    name: str
    value: Union[str, int, float, Dict]
    category: str  # color, spacing, typography, etc.
    tier: str  # primitive, semantic, component

    description: str = ""
    deprecated: bool = False
    replacement: Optional[str] = None


class TokenSystem:
    """
    Three-tier token architecture

    1. Primitive (Global): Raw values
    2. Semantic (Alias): Purpose-based references
    3. Component: Component-specific tokens
    """

    def __init__(self):
        self.tokens = {
            'primitive': {},
            'semantic': {},
            'component': {}
        }

    def add_primitive(self, name: str, value: str, category: str):
        """
        Primitive tokens: Raw design values
        Examples: blue-500, spacing-4, font-size-16
        """
        self.tokens['primitive'][name] = DesignToken(
            name=name,
            value=value,
            category=category,
            tier='primitive'
        )

    def add_semantic(self, name: str, reference: str, description: str = ""):
        """
        Semantic tokens: Purpose-based aliases
        Examples: color-primary, spacing-component-gap
        """
        # Semantic tokens reference primitives
        primitive = self.tokens['primitive'].get(reference)
        if not primitive:
            raise ValueError(f"Primitive token '{reference}' not found")

        self.tokens['semantic'][name] = DesignToken(
            name=name,
            value=f"{{primitive.{reference}}}",
            category=primitive.category,
            tier='semantic',
            description=description
        )

    def add_component_token(self, component: str, property: str, reference: str):
        """
        Component tokens: Component-specific values
        Examples: button-background-color, card-border-radius
        """
        name = f"{component}-{property}"
        self.tokens['component'][name] = DesignToken(
            name=name,
            value=f"{{semantic.{reference}}}",
            category='component',
            tier='component'
        )


# Example token system
tokens = TokenSystem()

# Primitives (raw values)
tokens.add_primitive('blue-500', '#0066cc', 'color')
tokens.add_primitive('blue-600', '#0052a3', 'color')
tokens.add_primitive('gray-100', '#f5f5f5', 'color')
tokens.add_primitive('gray-900', '#1a1a1a', 'color')
tokens.add_primitive('spacing-4', '16px', 'spacing')
tokens.add_primitive('spacing-6', '24px', 'spacing')
tokens.add_primitive('radius-md', '8px', 'border-radius')

# Semantic (purpose-based)
tokens.add_semantic('color-primary', 'blue-500', 'Primary brand color')
tokens.add_semantic('color-primary-hover', 'blue-600', 'Primary color on hover')
tokens.add_semantic('color-background', 'gray-100', 'Page background')
tokens.add_semantic('color-text', 'gray-900', 'Primary text color')
tokens.add_semantic('spacing-component-gap', 'spacing-4', 'Gap between components')

# Component-specific
tokens.add_component_token('button', 'background-color', 'color-primary')
tokens.add_component_token('button', 'border-radius', 'radius-md')
```

### Token Formats

```python
class TokenExporter:
    """Export tokens to various platforms"""

    def to_css_variables(self, tokens: Dict) -> str:
        """Export as CSS custom properties"""
        lines = [":root {"]
        for name, token in tokens.items():
            lines.append(f"  --{name}: {token.value};")
        lines.append("}")
        return "\n".join(lines)

    def to_scss_variables(self, tokens: Dict) -> str:
        """Export as SCSS variables"""
        lines = []
        for name, token in tokens.items():
            lines.append(f"${name}: {token.value};")
        return "\n".join(lines)

    def to_js_object(self, tokens: Dict) -> str:
        """Export as JavaScript/TypeScript object"""
        lines = ["export const tokens = {"]
        for name, token in tokens.items():
            js_name = name.replace('-', '_')
            lines.append(f"  {js_name}: '{token.value}',")
        lines.append("};")
        return "\n".join(lines)

    def to_json(self, tokens: Dict) -> str:
        """Export as JSON (for tools like Style Dictionary)"""
        import json
        output = {}
        for name, token in tokens.items():
            output[name] = {
                'value': token.value,
                'category': token.category,
                'description': token.description
            }
        return json.dumps(output, indent=2)


# Style Dictionary config example
STYLE_DICTIONARY_CONFIG = """
// config.json for Style Dictionary
{
  "source": ["tokens/**/*.json"],
  "platforms": {
    "css": {
      "transformGroup": "css",
      "buildPath": "dist/css/",
      "files": [{
        "destination": "variables.css",
        "format": "css/variables"
      }]
    },
    "js": {
      "transformGroup": "js",
      "buildPath": "dist/js/",
      "files": [{
        "destination": "tokens.js",
        "format": "javascript/es6"
      }]
    },
    "ios": {
      "transformGroup": "ios-swift",
      "buildPath": "dist/ios/",
      "files": [{
        "destination": "Tokens.swift",
        "format": "ios-swift/class.swift"
      }]
    }
  }
}
"""
```

## Component Architecture

### Component API Design

```typescript
// Well-designed component API
interface ButtonProps {
  // Visual variants
  variant: 'primary' | 'secondary' | 'ghost' | 'danger';
  size: 'sm' | 'md' | 'lg';

  // State
  disabled?: boolean;
  loading?: boolean;

  // Content
  children: React.ReactNode;
  leftIcon?: React.ReactNode;
  rightIcon?: React.ReactNode;

  // Behavior
  onClick?: (event: React.MouseEvent) => void;
  type?: 'button' | 'submit' | 'reset';

  // Accessibility
  'aria-label'?: string;

  // Extensibility
  className?: string;
  style?: React.CSSProperties;
}

// Component implementation
const Button: React.FC<ButtonProps> = ({
  variant = 'primary',
  size = 'md',
  disabled = false,
  loading = false,
  children,
  leftIcon,
  rightIcon,
  onClick,
  type = 'button',
  className,
  ...props
}) => {
  return (
    <button
      type={type}
      className={clsx(
        'btn',
        `btn--${variant}`,
        `btn--${size}`,
        loading && 'btn--loading',
        className
      )}
      disabled={disabled || loading}
      onClick={onClick}
      {...props}
    >
      {loading ? (
        <Spinner size="sm" />
      ) : (
        <>
          {leftIcon && <span className="btn__icon-left">{leftIcon}</span>}
          {children}
          {rightIcon && <span className="btn__icon-right">{rightIcon}</span>}
        </>
      )}
    </button>
  );
};
```

### Component Composition

```typescript
// Compound component pattern
const Card = {
  Root: ({ children, className, ...props }) => (
    <div className={clsx('card', className)} {...props}>
      {children}
    </div>
  ),

  Header: ({ children, className }) => (
    <div className={clsx('card__header', className)}>{children}</div>
  ),

  Body: ({ children, className }) => (
    <div className={clsx('card__body', className)}>{children}</div>
  ),

  Footer: ({ children, className }) => (
    <div className={clsx('card__footer', className)}>{children}</div>
  ),

  Image: ({ src, alt, className }) => (
    <img src={src} alt={alt} className={clsx('card__image', className)} />
  )
};

// Usage - flexible composition
<Card.Root>
  <Card.Image src="/product.jpg" alt="Product" />
  <Card.Header>
    <h3>Product Title</h3>
  </Card.Header>
  <Card.Body>
    <p>Product description...</p>
  </Card.Body>
  <Card.Footer>
    <Button>Add to Cart</Button>
  </Card.Footer>
</Card.Root>


// Render props pattern for complex components
interface DataTableProps<T> {
  data: T[];
  columns: Column<T>[];
  renderRow?: (item: T, index: number) => React.ReactNode;
  renderEmpty?: () => React.ReactNode;
}

const DataTable = <T,>({
  data,
  columns,
  renderRow,
  renderEmpty = () => <p>No data available</p>
}: DataTableProps<T>) => {
  if (data.length === 0) {
    return renderEmpty();
  }

  return (
    <table className="data-table">
      <thead>
        <tr>
          {columns.map(col => (
            <th key={col.key}>{col.header}</th>
          ))}
        </tr>
      </thead>
      <tbody>
        {data.map((item, index) =>
          renderRow ? renderRow(item, index) : (
            <tr key={index}>
              {columns.map(col => (
                <td key={col.key}>{col.render(item)}</td>
              ))}
            </tr>
          )
        )}
      </tbody>
    </table>
  );
};
```

### Variant Management

```typescript
// Using class-variance-authority (CVA)
import { cva, type VariantProps } from 'class-variance-authority';

const buttonVariants = cva(
  // Base styles
  'inline-flex items-center justify-center rounded-md font-medium transition-colors focus-visible:outline-none focus-visible:ring-2',
  {
    variants: {
      variant: {
        primary: 'bg-primary text-white hover:bg-primary-dark',
        secondary: 'bg-secondary text-gray-900 hover:bg-secondary-dark',
        ghost: 'bg-transparent hover:bg-gray-100',
        danger: 'bg-red-600 text-white hover:bg-red-700',
      },
      size: {
        sm: 'h-8 px-3 text-sm',
        md: 'h-10 px-4 text-base',
        lg: 'h-12 px-6 text-lg',
      },
    },
    defaultVariants: {
      variant: 'primary',
      size: 'md',
    },
  }
);

interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {}

const Button: React.FC<ButtonProps> = ({
  variant,
  size,
  className,
  ...props
}) => {
  return (
    <button
      className={buttonVariants({ variant, size, className })}
      {...props}
    />
  );
};
```

## Storybook Documentation

### Story Structure

```typescript
// Button.stories.tsx
import type { Meta, StoryObj } from '@storybook/react';
import { Button } from './Button';

const meta: Meta<typeof Button> = {
  title: 'Components/Button',
  component: Button,
  tags: ['autodocs'],
  argTypes: {
    variant: {
      control: 'select',
      options: ['primary', 'secondary', 'ghost', 'danger'],
      description: 'Visual style variant',
      table: {
        defaultValue: { summary: 'primary' },
      },
    },
    size: {
      control: 'radio',
      options: ['sm', 'md', 'lg'],
    },
    disabled: {
      control: 'boolean',
    },
    onClick: { action: 'clicked' },
  },
  parameters: {
    docs: {
      description: {
        component: 'Primary UI component for user interaction.',
      },
    },
  },
};

export default meta;
type Story = StoryObj<typeof Button>;

// Primary variant
export const Primary: Story = {
  args: {
    variant: 'primary',
    children: 'Primary Button',
  },
};

// All variants
export const AllVariants: Story = {
  render: () => (
    <div style={{ display: 'flex', gap: '1rem' }}>
      <Button variant="primary">Primary</Button>
      <Button variant="secondary">Secondary</Button>
      <Button variant="ghost">Ghost</Button>
      <Button variant="danger">Danger</Button>
    </div>
  ),
};

// Sizes
export const Sizes: Story = {
  render: () => (
    <div style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
      <Button size="sm">Small</Button>
      <Button size="md">Medium</Button>
      <Button size="lg">Large</Button>
    </div>
  ),
};

// States
export const States: Story = {
  render: () => (
    <div style={{ display: 'flex', gap: '1rem' }}>
      <Button>Default</Button>
      <Button disabled>Disabled</Button>
      <Button loading>Loading</Button>
    </div>
  ),
};

// With icons
export const WithIcons: Story = {
  render: () => (
    <div style={{ display: 'flex', gap: '1rem' }}>
      <Button leftIcon={<PlusIcon />}>Add Item</Button>
      <Button rightIcon={<ArrowRightIcon />}>Continue</Button>
    </div>
  ),
};
```

### Storybook Configuration

```javascript
// .storybook/main.ts
import type { StorybookConfig } from '@storybook/react-vite';

const config: StorybookConfig = {
  stories: ['../src/**/*.stories.@(js|jsx|ts|tsx|mdx)'],
  addons: [
    '@storybook/addon-links',
    '@storybook/addon-essentials',
    '@storybook/addon-interactions',
    '@storybook/addon-a11y',  // Accessibility addon
    '@storybook/addon-designs',  // Figma integration
  ],
  framework: {
    name: '@storybook/react-vite',
    options: {},
  },
  docs: {
    autodocs: 'tag',
  },
};

export default config;

// .storybook/preview.ts
import type { Preview } from '@storybook/react';
import '../src/styles/globals.css';

const preview: Preview = {
  parameters: {
    actions: { argTypesRegex: '^on[A-Z].*' },
    controls: {
      matchers: {
        color: /(background|color)$/i,
        date: /Date$/,
      },
    },
    // Viewport presets
    viewport: {
      viewports: {
        mobile: { name: 'Mobile', styles: { width: '375px', height: '667px' } },
        tablet: { name: 'Tablet', styles: { width: '768px', height: '1024px' } },
        desktop: { name: 'Desktop', styles: { width: '1440px', height: '900px' } },
      },
    },
    // Background options
    backgrounds: {
      default: 'light',
      values: [
        { name: 'light', value: '#ffffff' },
        { name: 'dark', value: '#1a1a1a' },
        { name: 'gray', value: '#f5f5f5' },
      ],
    },
  },
};

export default preview;
```

### MDX Documentation

```mdx
{/* Button.mdx */}
import { Meta, Story, Canvas, ArgsTable } from '@storybook/blocks';
import { Button } from './Button';
import * as ButtonStories from './Button.stories';

<Meta of={ButtonStories} />

# Button

Buttons are used to trigger actions or navigate to new pages.

## Usage Guidelines

### When to use
- Triggering form submissions
- Initiating actions (delete, save, etc.)
- Navigation to important pages

### When NOT to use
- For navigation to regular pages (use links)
- For toggles (use switches)
- For selection (use checkboxes/radios)

## Variants

<Canvas of={ButtonStories.AllVariants} />

### Primary
Use for the main action on a page. Limit to one per section.

### Secondary
Use for secondary actions that complement the primary action.

### Ghost
Use for tertiary actions or in compact spaces.

### Danger
Use for destructive actions like delete.

## Sizes

<Canvas of={ButtonStories.Sizes} />

| Size | Use Case |
|------|----------|
| Small | Compact UIs, tables, inline actions |
| Medium | Default for most cases |
| Large | Hero sections, emphasized actions |

## Props

<ArgsTable of={Button} />

## Accessibility

- Always include descriptive text or `aria-label`
- Ensure sufficient color contrast (4.5:1 minimum)
- Focus indicator must be visible
- Loading state should announce to screen readers
```

## Theming

### Theme Structure

```typescript
// Theme type definition
interface Theme {
  colors: {
    primary: string;
    primaryHover: string;
    secondary: string;
    background: string;
    surface: string;
    text: string;
    textMuted: string;
    border: string;
    error: string;
    success: string;
  };
  spacing: {
    xs: string;
    sm: string;
    md: string;
    lg: string;
    xl: string;
  };
  typography: {
    fontFamily: string;
    fontSize: {
      xs: string;
      sm: string;
      md: string;
      lg: string;
      xl: string;
    };
    fontWeight: {
      normal: number;
      medium: number;
      bold: number;
    };
  };
  borderRadius: {
    sm: string;
    md: string;
    lg: string;
    full: string;
  };
  shadows: {
    sm: string;
    md: string;
    lg: string;
  };
}

// Light theme
const lightTheme: Theme = {
  colors: {
    primary: '#0066cc',
    primaryHover: '#0052a3',
    secondary: '#6c757d',
    background: '#ffffff',
    surface: '#f8f9fa',
    text: '#212529',
    textMuted: '#6c757d',
    border: '#dee2e6',
    error: '#dc3545',
    success: '#28a745',
  },
  // ... rest of theme
};

// Dark theme
const darkTheme: Theme = {
  colors: {
    primary: '#4da6ff',
    primaryHover: '#80bfff',
    secondary: '#adb5bd',
    background: '#121212',
    surface: '#1e1e1e',
    text: '#e9ecef',
    textMuted: '#adb5bd',
    border: '#495057',
    error: '#f87171',
    success: '#4ade80',
  },
  // ... rest of theme
};

// Theme provider (React)
import { createContext, useContext, useState } from 'react';

const ThemeContext = createContext<{
  theme: Theme;
  toggleTheme: () => void;
}>({
  theme: lightTheme,
  toggleTheme: () => {},
});

export const ThemeProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [isDark, setIsDark] = useState(false);

  return (
    <ThemeContext.Provider
      value={{
        theme: isDark ? darkTheme : lightTheme,
        toggleTheme: () => setIsDark(!isDark),
      }}
    >
      {children}
    </ThemeContext.Provider>
  );
};

export const useTheme = () => useContext(ThemeContext);
```

### CSS Variable Theming

```css
/* Theme tokens as CSS variables */
:root {
  /* Light theme (default) */
  --color-primary: #0066cc;
  --color-primary-hover: #0052a3;
  --color-background: #ffffff;
  --color-surface: #f8f9fa;
  --color-text: #212529;
  --color-text-muted: #6c757d;
  --color-border: #dee2e6;
}

[data-theme="dark"] {
  --color-primary: #4da6ff;
  --color-primary-hover: #80bfff;
  --color-background: #121212;
  --color-surface: #1e1e1e;
  --color-text: #e9ecef;
  --color-text-muted: #adb5bd;
  --color-border: #495057;
}

/* System preference support */
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --color-primary: #4da6ff;
    --color-background: #121212;
    /* ... dark theme values */
  }
}

/* Components use tokens */
.button {
  background-color: var(--color-primary);
  color: var(--color-background);
}

.button:hover {
  background-color: var(--color-primary-hover);
}

.card {
  background-color: var(--color-surface);
  border: 1px solid var(--color-border);
  color: var(--color-text);
}
```

## Governance & Contribution

### Contribution Process

```python
CONTRIBUTION_WORKFLOW = {
    'proposal': {
        'step': 1,
        'description': "Submit RFC for new component/pattern",
        'template': """
## Component Proposal: [Name]

### Problem Statement
What problem does this solve?

### Proposed Solution
Brief description of the component.

### API Design
```tsx
interface ComponentProps {
  // Proposed props
}
```

### Alternatives Considered
What other approaches were considered?

### Design Reference
Link to Figma/design specs
        """
    },
    'review': {
        'step': 2,
        'description': "Design system team reviews proposal",
        'criteria': [
            "Solves common problem (used by 3+ products)",
            "Consistent with existing patterns",
            "Accessible by default",
            "Well-documented API"
        ]
    },
    'implementation': {
        'step': 3,
        'description': "Build component following standards",
        'requirements': [
            "Component implementation",
            "Unit tests (90%+ coverage)",
            "Storybook stories",
            "Documentation",
            "Accessibility audit"
        ]
    },
    'release': {
        'step': 4,
        'description': "Merge and release",
        'process': [
            "PR review by 2+ maintainers",
            "Accessibility review",
            "Visual regression tests pass",
            "Changelog entry",
            "Version bump"
        ]
    }
}
```

### Versioning Strategy

```python
VERSIONING_GUIDELINES = {
    'major': {
        'when': [
            "Breaking API changes",
            "Removing components",
            "Changing token structure"
        ],
        'process': "Migration guide required, deprecation period"
    },
    'minor': {
        'when': [
            "New components",
            "New variants/props",
            "New tokens (non-breaking)"
        ],
        'process': "Standard release, changelog entry"
    },
    'patch': {
        'when': [
            "Bug fixes",
            "Documentation updates",
            "Performance improvements"
        ],
        'process': "Fast-track release"
    }
}

# Deprecation policy
DEPRECATION_POLICY = """
1. Mark as deprecated in current minor version
2. Log console warning when used
3. Document migration path
4. Remove in next major version (minimum 3 months)
"""
```

## Best Practices

1. **Start with tokens**: Foundation before components
2. **Design for composition**: Small, composable pieces
3. **Accessibility first**: Built-in, not bolted-on
4. **Document everything**: API, usage, dos/don'ts
5. **Version carefully**: Breaking changes hurt adoption
6. **Gather feedback**: Regular surveys and usage analytics

## Common Pitfalls

- **Over-abstraction**: Components too generic to be useful
- **Under-documentation**: Team doesn't know how to use it
- **Ignoring adoption**: Building without user research
- **No governance**: Wild west of contributions
- **Token sprawl**: Too many tokens, no hierarchy
- **Platform inconsistency**: Different implementations drift apart

---

**Skill Type**: UX - Design Systems
**Complexity**: Advanced
**Typical Usage**: Component library development, team scaling
