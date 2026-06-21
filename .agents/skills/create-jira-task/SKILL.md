---
name: create-jira-task
description: Create or prepare Jira task tickets from user requests, bug reports, implementation notes, specs, or code-review findings. Use when Codex is asked to create a Jira task, draft a Jira issue, turn work into a ticket, prepare backlog items, or format work for Jira.
---

# Create Jira Task

## Overview

Turn a request into a clear Jira task with enough context for another engineer or product teammate to act on it. If a Jira/Atlassian tool is available and authorized, use it only after the required fields are clear; otherwise produce a copy-ready Jira draft.

## Workflow

1. Identify whether the user wants a live Jira issue created or only a draft.
2. Check available tools for Jira or Atlassian access. Do not assume Jira access exists.
3. Extract known fields from the request.
4. Ask for missing required fields only when they block creation. If drafting, use sensible placeholders instead.
5. Produce a concise ticket in Jira-friendly Markdown.
6. Before creating a live issue, confirm the target project, issue type, summary, and description unless the user already provided them and explicitly requested immediate creation.
7. After live creation, report the issue key/link and the exact fields used.

## Ticket Fields

Prefer these fields when available:

- Project
- Issue type, usually Task, Bug, Story, or Spike
- Summary
- Description
- Acceptance criteria
- Priority
- Labels
- Components
- Assignee
- Parent, epic, or sprint
- Links to relevant specs, PRs, commits, screenshots, logs, or files

## Draft Format

Use this structure for drafts:

```md
Project: <project key or name>
Issue type: <Task | Bug | Story | Spike>
Priority: <priority or TBD>
Labels: <labels or none>

Summary:
<short imperative or outcome-focused title>

Description:
<context, current behavior, desired behavior, and relevant constraints>

Acceptance criteria:
- <observable outcome>
- <observable outcome>

Implementation notes:
- <optional technical notes>

References:
- <SPEC.md, file path, issue link, PR, screenshot, or log>
```

## Quality Bar

- Keep the summary short and action-oriented.
- Keep the description useful without overloading it with implementation detail.
- Write acceptance criteria as verifiable outcomes.
- Preserve user-provided project names, priorities, labels, and wording when they appear intentional.
- Do not invent customer impact, severity, dates, or commitments.
- For Poly Chat tasks, call out any contradiction with `SPEC.md`.

## Live Jira Creation

When a Jira tool is available:

1. Read the tool schema before calling it.
2. Map draft fields to the tool's required fields.
3. Use the smallest safe set of fields needed to create the issue.
4. If creation fails, report the error and provide the draft so the user can still proceed.
