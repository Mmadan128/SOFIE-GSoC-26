---
title: "Logbook"
layout: single
permalink: /logbook/
author_profile: false
---

Project updates are published as separate dated posts.

## Entries

{% if site.posts.size > 0 %}
<table>
	<thead>
		<tr>
			<th>Date</th>
			<th>Entry</th>
			<th>Summary</th>
		</tr>
	</thead>
	<tbody>
{% for post in site.posts %}
		<tr>
			<td>{{ post.date | date: "%Y-%m-%d" }}</td>
			<td><a href="{{ post.url | relative_url }}">{{ post.title }}</a></td>
			<td>{{ post.summary | default: post.excerpt | strip_html | truncate: 130 }}</td>
		</tr>
{% endfor %}
	</tbody>
</table>
{% else %}
No entries yet. Add your first file under `_posts/`.
{% endif %}
