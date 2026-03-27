---
title: "Weekly Engineering Logbook"
layout: single
permalink: /logbook/
author_profile: false
---

This page is the index for weekly engineering entries. Each update should be a separate post in `_posts/`.

## Why Post-Based Logbook

- One file per week keeps history clean and reviewable.
- Git diffs stay small and easier to audit.
- You can include full code blocks, terminal output, and diagrams naturally.

## Entry Template

Create a new file in `_posts/` using:

`YYYY-MM-DD-short-title.md`

Use this structure in each post:

```markdown
---
title: "Week N - Topic"
layout: single
author_profile: false
---

## Planned
- 

## Completed
- 

## Evidence
- 

## Code Notes
```cpp
// Example kernel snippet
ALPAKA_FN_ACC void operator()(Acc const& acc) const {
	// implementation notes
}
```

## Risks or Blockers
- 

## Next Actions
- 
```

## Entries

{% if site.posts.size > 0 %}
{% for post in site.posts %}
- **[{{ post.title }}]({{ post.url | relative_url }})** - {{ post.date | date: "%Y-%m-%d" }}
	{{ post.excerpt | strip_html | truncate: 170 }}
{% endfor %}
{% else %}
No entries yet. Add your first file under `_posts/`.
{% endif %}
