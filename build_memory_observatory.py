#!/usr/bin/env python3
"""Build The Memory Observatory.

Generates a synthetic, narratively anchored 120-day team conversation
(five people, 1 Jan to 30 Apr 2026), replays T-Mem's query-time retrieval
on it with a toy BM25, DBSCAN over timestamps and hand-annotated labels,
and writes:

  memory_kg.json    the conversation, its entity graph and the replay
  docs/index.html   template.html with the data inlined; GitHub Pages serves docs/

The retrieval stages follow arXiv 2610.12874 (T-Mem): lexical seeds,
temporal evidence discovery, Key / Support / Irrelevant labels, Update and
Expand links, an 8,192-token forest budget. No model is called; the labels
that T-Mem's LLM would produce are annotated in this script.
"""
import datetime
import json
import math
import pathlib
import random
import re

HERE = pathlib.Path(__file__).resolve().parent
rng = random.Random(20261013)
START = datetime.date(2026, 1, 1)
C = 120
BUDGET = 8192
QUERY = "What did we agree about the auth refactor in March?"


def date(d):
    return START + datetime.timedelta(days=int(d))


def dlabel(d):
    return date(d).strftime("%-d %b")


def weekend(d):
    return date(d).weekday() >= 5


# ---------------------------------------------------------------- entities
E = {}


def ent(id, name, label, desc):
    E[id] = dict(id=id, name=name, label=label, description=desc)


ent("dana", "Dana Whitford", "PERSON", "Engineering lead. Runs the design reviews and makes the calls.")
ent("priya", "Priya Raman", "PERSON", "Backend engineer. Owns the auth service and the auth_v2 flag.")
ent("tomas", "Tomás Ferreira", "PERSON", "Infrastructure. Owns the staging cluster, Postgres and the pager.")
ent("mei", "Mei Lin", "PERSON", "Frontend. Owns the client side of the token refresh contract and the tenant dashboard.")
ent("jonah", "Jonah Adler", "PERSON", "Product. Writes the release plan and the tenant notices.")
PEOPLE = ["dana", "priya", "tomas", "mei", "jonah"]
ent("auth_service", "Auth service", "SYSTEM", "Issues and refreshes tokens; keeps sessions in the session store.")
ent("session_store", "Session store", "SYSTEM", "Where live sessions are kept. Runs on Redis.")
ent("redis", "Redis", "SYSTEM", "In-memory store behind sessions. p99 of 2 ms.")
ent("postgres", "Postgres", "SYSTEM", "The primary database. Upgraded on staging in mid March.")
ent("token_refresh", "Token refresh path", "SYSTEM", "The server side of /auth/refresh. Rewritten in the auth refactor.")
ent("auth_v2", "auth_v2 flag", "SYSTEM", "Feature flag guarding the rewritten refresh path. Rollback is flipping it off.")
ent("staging_cluster", "Staging cluster", "SYSTEM", "Pre-production cluster. Took the Postgres upgrade the week of 16 March.")
ent("ci", "CI pipeline", "SYSTEM", "Builds and tests every PR. Flaky in January.")
ent("search_index", "Search index", "SYSTEM", "Hybrid index behind tenant search.")
ent("reranker", "Reranker", "SYSTEM", "Cross-encoder rerank stage. Capped at 40 candidates after INC-2041.")
ent("query_planner", "Query planner", "SYSTEM", "Chooses the retrieval plan per query.")
ent("tenant_dash", "Tenant dashboard", "SYSTEM", "The customer-facing dashboard shipped in 2.3.")
ent("pager", "On-call pager", "SYSTEM", "Pages whoever is on call.")
ent("design_doc", "Auth refactor design doc", "DOCUMENT", "Priya's proposal, posted 9 March, revised after the review.")
ent("tenant_notice", "Tenant notice", "DOCUMENT", "Customer notice for the auth cutover, drafted by Jonah.")
ent("design_review", "Design review", "MEETING", "Where the auth refactor was decided, 10 March.")
ent("retro", "Retro", "MEETING", "Monthly retrospective.")
ent("planning", "Planning", "MEETING", "Quarterly planning.")
ent("dec_park", "Park the auth refactor until after 2.3", "DECISION", "January decision to wait.")
ent("dec_keep_redis", "Keep Redis for sessions", "DECISION", "Decided at the design review on 10 March.")
ent("dec_cut16", "Cutover on 16 March", "DECISION", "The first cutover date, decided 10 March.")
ent("dec_cut20", "Cutover on 20 March", "DECISION", "Replaced 16 March on 11 March because of the staging upgrade.")
ent("dec_rerank", "Cap the reranker at 40 candidates", "DECISION", "Decided after the search latency incident.")
ent("dec_hire", "Open a second backend role", "DECISION", "Decided at planning.")
ent("rel23", "Release 2.3", "RELEASE", "The spring release: tenant dashboard, search fixes.")
ent("rel221", "Release 2.2.1", "RELEASE", "Hotfix release after the pager storm.")
ent("inc2041", "INC-2041 search timeouts", "INCIDENT", "Search p95 over 800 ms, traced to the reranker.")
ent("inc2057", "INC-2057 pager storm", "INCIDENT", "Alert loop that paged Tomás 41 times in one night.")
ent("inc2066", "INC-2066 staging outage", "INCIDENT", "Staging down for an afternoon during the Postgres upgrade rehearsal.")
ent("inc2079", "INC-2079 refresh errors", "INCIDENT", "Token refresh errors for one tenant after a config change in April.")
ent("backend_role", "Backend role", "ROLE", "The second backend opening.")
ent("candidate", "The backend candidate", "PERSON", "Interviewed by Priya and Jonah in early April.")

# ---------------------------------------------------------------- messages
MSGS = []
APPS = {k: set() for k in E}
RELS = {}


def say(d, hour, who, text, topic, ents=(), rels=(), key=False, support=False, tag=None):
    ents = set(ents) | {who}
    m = dict(d=round(d + hour / 24, 4), who=who, text=text, topic=topic,
             ents=ents, key=key, support=support, tag=tag)
    MSGS.append(m)
    for s, l, t in rels:
        s = who if s == "{who}" else s
        r = RELS.setdefault((s, l, t), dict(first=m["d"], apps=set()))
        r["first"] = min(r["first"], m["d"])
        r["apps"].add(d)
        ents.update((s, t))
    for e in ents:
        APPS[e].add(d)
    m["ents"] = sorted(ents)
    return m


def h(lo=9.0, hi=18.0):
    return rng.uniform(lo, hi)


CHORES = [
    ("chasing the CI pipeline flake", ["ci"]), ("tenant dashboard filters", ["tenant_dash"]),
    ("search index backfill", ["search_index"]), ("on-call handover", ["pager"]),
    ("retro notes", ["retro"]), ("PR reviews", []), ("1:1s all afternoon", []),
    ("release notes for 2.3", ["rel23"]), ("query planner tests", ["query_planner"]),
    ("staging cluster patching", ["staging_cluster"]), ("Postgres vacuum tuning", ["postgres"]),
    ("the reranker eval set", ["reranker"]), ("pager dedupe rules", ["pager"]),
    ("tenant dashboard a11y pass", ["tenant_dash"]), ("interview prep", []),
    ("writing the planning doc", ["planning"]), ("session store metrics", ["session_store"]),
]
PRS = [
    ("tighten CI pipeline retries", ["ci"]), ("tenant dashboard empty state", ["tenant_dash"]),
    ("search index chunk size", ["search_index"]), ("query planner fallback", ["query_planner"]),
    ("pager dedupe", ["pager"]), ("staging cluster secrets rotation", ["staging_cluster"]),
    ("Postgres connection pool", ["postgres"]), ("reranker batch size", ["reranker"]),
    ("release 2.3 changelog", ["rel23"]), ("session store TTL cleanup", ["session_store"]),
    ("dashboard export to CSV", ["tenant_dash"]), ("flaky test quarantine", ["ci"]),
]
MISC = [
    ("lunch?", []), ("wfh today", []), ("coffee machine is dead again", []), ("back at 2, dentist", []),
    ("anyone else seeing a slow VPN", []), ("who has the staging password", ["staging_cluster"]),
    ("standup moved to 10:15", []), ("leaving early, kid pickup", []), ("the new build is green", ["ci"]),
    ("I am on the pager tonight", ["pager"]), ("thanks for the review", []), ("merged", []),
    ("can someone restart the search index worker", ["search_index"]), ("+1", []),
    ("notes from the customer call are in the drive", []), ("ok", []), ("on it", []),
    ("dashboard demo at 3 if anyone wants to watch", ["tenant_dash"]),
]


def filler(d):
    if weekend(d):
        for _ in range(rng.randint(0, 2)):
            t, e = rng.choice(MISC)
            say(d, h(10, 20), rng.choice(PEOPLE), t, "chatter", e)
        return
    for who in rng.sample(PEOPLE, rng.randint(4, 5)):
        a, ea = rng.choice(CHORES)
        b, eb = rng.choice(CHORES)
        say(d, h(9.4, 10.2), who, f"standup: yesterday {a}, today {b}", "standup", ea + eb)
    for _ in range(rng.randint(3, 6)):
        t, e = rng.choice(PRS)
        say(d, h(10, 17.5), rng.choice(PEOPLE), f"PR #{rng.randint(1200, 1990)}: {t}, ready for review", "prs", e)
    for _ in range(rng.randint(8, 14)):
        t, e = rng.choice(MISC)
        say(d, h(8.5, 18.5), rng.choice(PEOPLE), t, "chatter", e)


# thread templates: (speaker or None, text, entity ids, relations)
SEARCH = [
    ("tomas", "p95 on the search index is {ms} ms again. The reranker is the suspect.", ["search_index", "reranker"], [("search_index", "SLOWED_BY", "reranker")]),
    ("priya", "query planner picked the slow plan for the tenant filter case", ["query_planner"], [("query_planner", "PART_OF", "search_index")]),
    ("tomas", "I can reproduce the timeouts with 200 candidates into the reranker", ["reranker"], [("{who}", "INVESTIGATES", "search_index")]),
    ("priya", "capping the reranker at 40 candidates brings p95 to 310 ms", ["reranker"], [("dec_rerank", "CAPS", "reranker")]),
    ("mei", "search latency dashboard updated with the rerank stage", ["search_index"], []),
    ("dana", "let's take the cap to the retro and decide there", ["dec_rerank", "retro"], []),
    (None, "backfill of the search index finished for the last three tenants", ["search_index"], []),
    (None, "the query planner fallback is in, waiting on CI", ["query_planner", "ci"], []),
    ("jonah", "two tenants asked about search speed this week", ["search_index"], []),
    (None, "rerank eval numbers in the doc, 40 candidates loses 1 point of recall", ["reranker"], []),
]
RELEASE = [
    ("jonah", "2.3 scope check: tenant dashboard, search fixes, nothing else", ["rel23", "tenant_dash"], [("rel23", "INCLUDES", "tenant_dash")]),
    ("tomas", "CI pipeline is green on the 2.3 branch", ["ci", "rel23"], [("rel23", "BUILT_BY", "ci")]),
    ("mei", "tenant dashboard empty state is in, screenshots in the PR", ["tenant_dash"], [("{who}", "SHIPS", "tenant_dash")]),
    ("jonah", "release notes draft for 2.3, comments welcome", ["rel23"], [("{who}", "TRACKS", "rel23")]),
    ("dana", "2.3 go or no go on Thursday", ["rel23"], []),
    (None, "2.3 branch cut, only fixes from here", ["rel23"], []),
    ("mei", "dashboard export to CSV is in 2.3 after all", ["tenant_dash", "rel23"], []),
    (None, "staging has the 2.3 candidate, please poke at it", ["staging_cluster", "rel23"], []),
    ("priya", "session store TTL change is in the 2.3 candidate", ["session_store", "rel23"], []),
]
RELEASE_FEB = RELEASE + [
    ("jonah", "if 2.3 slips into March we lose the Q1 number", ["rel23"], []),
    ("tomas", "hotfix 2.2.1 is out, the pager storm is fixed", ["rel221", "inc2057"], [("rel221", "HOTFIX_FOR", "inc2057")]),
]
RELEASE_APR = RELEASE + [
    ("jonah", "2.3 goes to 10% of tenants Monday, 50% Wednesday, everyone Friday", ["rel23"], []),
    ("priya", "auth_v2 is on for 10% of tenants with 2.3, refresh errors flat", ["auth_v2", "rel23"], [("auth_v2", "SHIPPED_IN", "rel23")]),
    ("mei", "tenant dashboard is live for everyone, no complaints yet", ["tenant_dash"], []),
]
HIRING_FEB = [
    ("dana", "planning outcome: we open a second backend role", ["planning", "backend_role", "dec_hire"], [("planning", "DECIDES", "dec_hire"), ("dec_hire", "OPENS", "backend_role")]),
    ("jonah", "job post for the backend role is live", ["backend_role"], []),
    (None, "please share the backend role post", ["backend_role"], []),
    ("dana", "interview loop for the backend role: systems, product sense, values", ["backend_role"], []),
]
HIRING_APR = [
    ("dana", "the backend candidate's take-home is in, Priya can you review", ["candidate"], [("priya", "INTERVIEWS", "candidate")]),
    ("jonah", "candidate loop Thursday: Priya on systems, me on product sense", ["candidate"], [("{who}", "INTERVIEWS", "candidate")]),
    ("priya", "take-home is solid, a few questions about the error handling", ["candidate"], []),
    ("dana", "debrief notes for the candidate are in the drive", ["candidate"], []),
    ("jonah", "offer draft for the backend role with the candidate", ["candidate", "backend_role"], [("candidate", "CONSIDERED_FOR", "backend_role")]),
]
INCIDENTS = {
    19: [("tomas", "INC-2041 opened: search timeouts for three tenants", ["inc2041", "search_index"], [("inc2041", "HIT", "search_index")]),
         ("priya", "rolled the reranker back to the old batch size", ["reranker"], [("inc2041", "CAUSED_BY", "reranker")]),
         ("tomas", "tenants recovered, keeping INC-2041 open for the postmortem", ["inc2041"], []),
         ("dana", "postmortem Friday, no blame, bring the graphs", ["inc2041"], [])],
    47: [("tomas", "INC-2057: the pager paged me 41 times overnight", ["inc2057", "pager"], [("inc2057", "PAGED", "tomas")]),
         ("tomas", "alert loop was the CI pipeline health check", ["ci", "inc2057"], [("inc2057", "CAUSED_BY", "ci")]),
         ("dana", "muting the check until the dedupe lands", ["pager"], []),
         ("priya", "pager dedupe PR is up, sorry Tomás", ["pager"], [])],
    61: [("tomas", "INC-2066: staging cluster is down, the Postgres upgrade rehearsal went sideways", ["inc2066", "staging_cluster", "postgres"], [("inc2066", "HIT", "staging_cluster")]),
         ("tomas", "restored from the snapshot, rehearsal again next week", ["staging_cluster"], []),
         ("mei", "no staging all afternoon then, I will test locally", ["staging_cluster"], []),
         ("dana", "INC-2066 closed, add the rehearsal to the calendar", ["inc2066"], [])],
    97: [("priya", "INC-2079: token refresh errors for one tenant since the config change", ["inc2079", "token_refresh"], [("inc2079", "HIT", "token_refresh")]),
         ("priya", "not the new path, the flag is still off for them. A rotated secret.", ["auth_v2"], []),
         ("tomas", "secret re-issued, errors gone", ["inc2079"], []),
         ("dana", "INC-2079 closed", ["inc2079"], [])],
}


def episode(d, templates, n, topic):
    for _ in range(n):
        who, text, ents, rels = rng.choice(templates)
        who = who or rng.choice(PEOPLE)
        text = text.replace("{ms}", str(rng.choice([640, 720, 810, 840, 910])))
        say(d, h(), who, text, topic, ents, rels)


EPISODES = [
    ("search", range(32, 37), SEARCH, (7, 10)),      # 2 to 6 Feb
    ("search", range(82, 85), SEARCH, (6, 8)),       # 24 to 26 Mar
    ("release", range(25, 30), RELEASE, (7, 10)),    # 26 to 30 Jan
    ("release", range(53, 58), RELEASE_FEB, (7, 10)),  # 23 to 27 Feb
    ("release", range(109, 114), RELEASE_APR, (7, 11)),  # 20 to 24 Apr
    ("hiring", range(40, 43), HIRING_FEB, (4, 6)),   # 10 to 12 Feb
    ("hiring", range(88, 92), HIRING_APR, (4, 6)),   # 30 Mar to 2 Apr
]


def search_decisions():
    say(36, 16.5, "dana", "Cap the reranker at 40 candidates and revisit after 2.3. Agreed?", "search",
        ["dec_rerank", "reranker"], [("{who}", "DECIDES", "dec_rerank"), ("dec_rerank", "CAPS", "reranker")])
    say(36, 16.55, "priya", "Agreed.", "search", [], [])
    say(36, 16.6, "tomas", "Agreed, and I will add the alert at 60.", "search", ["pager"], [])
    say(84, 11.2, "dana", "The 40 cap stays through 2.3, as agreed at the retro.", "search", ["dec_rerank", "retro"], [])


def auth_thread():
    # 13 January: the idea, parked
    say(12, 14.33, "priya", "Can we refactor the auth service's session handling after 2.3? It is held together with tape.", "auth",
        ["auth_service", "session_store"], [("priya", "OWNS", "auth_service"), ("auth_service", "USES", "session_store")], tag="jan_idea")
    say(12, 14.52, "dana", "Parked until after the release. Agreed in principle, not in schedule.", "auth",
        ["dec_park"], [("{who}", "DECIDES", "dec_park"), ("dec_park", "PARKS", "auth_service")], tag="jan_park")
    say(12, 14.58, "tomas", "If it touches Redis, loop me in.", "auth", [], [("session_store", "RUNS_ON", "redis")])
    # 9 March: the proposal
    say(67, 9.5, "priya", "Design doc for the auth refactor is in the drive. Proposal: move sessions off Redis onto Postgres and rewrite the token refresh path.", "auth",
        ["design_doc", "redis", "postgres", "token_refresh"], [("{who}", "WRITES", "design_doc"), ("design_doc", "PROPOSES", "postgres"), ("token_refresh", "PART_OF", "auth_service")], support=True, tag="doc")
    say(67, 10.08, "tomas", "Against moving sessions to Postgres. Redis p99 is 2 ms and the session writes would land on the primary.", "auth",
        ["redis", "postgres"], [("{who}", "OBJECTS_TO", "postgres")], support=True, tag="objection")
    say(67, 10.3, "priya", "The point is one fewer system to run. But I hear you on the primary.", "auth", ["postgres"], [], support=True)
    say(67, 10.5, "mei", "Does the token refresh rewrite change the frontend contract? We cache the refresh token in memory.", "auth",
        ["token_refresh"], [("{who}", "ASKS_ABOUT", "token_refresh")], support=True, tag="mei_q")
    say(67, 10.7, "priya", "Contract stays. The endpoint stays /auth/refresh, only the server side changes.", "auth", ["token_refresh"], [], support=True)
    say(67, 11.0, "dana", "Design review tomorrow at 11:00, the auth refactor only. Read the doc before.", "auth",
        ["design_review", "design_doc"], [("design_review", "REVIEWS", "design_doc")], support=True, tag="sched")
    say(67, 16.17, "jonah", "Will customers notice anything? If not, no notice needed.", "auth", [], [], support=True)
    say(67, 16.23, "priya", "They should not, but sessions might be invalidated at cutover. A notice is cheap.", "auth", ["session_store"], [], support=True)
    # 10 March: the review
    say(68, 11.67, "dana", "Design review outcome, so we all have it in one place: we agreed to keep Redis for sessions and rewrite only the token refresh path. Cutover 16 March behind a flag.", "auth",
        ["design_review", "dec_keep_redis", "dec_cut16", "redis", "token_refresh"],
        [("design_review", "DECIDES", "dec_keep_redis"), ("design_review", "DECIDES", "dec_cut16"), ("dec_keep_redis", "KEEPS", "redis")], key=True, tag="outcome")
    say(68, 11.87, "priya", "Agreed. The flag is auth_v2, off by default. Rollback is flipping it off.", "auth",
        ["auth_v2", "token_refresh"], [("auth_v2", "GUARDS", "token_refresh"), ("{who}", "OWNS", "auth_v2")], key=True, tag="flag")
    say(68, 11.92, "tomas", "Fine with me as long as sessions stay in Redis.", "auth", ["redis", "session_store"], [], support=True)
    say(68, 12.17, "jonah", "I will draft a tenant notice for the 16th anyway.", "auth", ["tenant_notice"], [("{who}", "DRAFTS", "tenant_notice")], support=True, tag="notice_q")
    say(68, 12.5, "mei", "Frontend needs nothing for auth_v2 then. I will add the flag to the smoke tests.", "auth", ["auth_v2"], [("{who}", "TESTS", "auth_v2")], support=True)
    say(68, 15.0, "priya", "Doc updated with the review outcome: Redis stays, refresh path rewrite, auth_v2 flag.", "auth", ["design_doc", "auth_v2"], [], support=True)
    # 11 March: the date moves
    say(69, 15.08, "tomas", "Staging cannot take the auth cutover on the 16th. The Postgres upgrade on the staging cluster is that week.", "auth",
        ["staging_cluster", "postgres"], [("{who}", "OWNS", "staging_cluster"), ("postgres", "UPGRADED_ON", "staging_cluster")], support=True, tag="staging_block")
    say(69, 15.33, "dana", "Then the cutover moves to 20 March. Same scope, same flag. Updating the doc.", "auth",
        ["dec_cut20", "dec_cut16", "design_doc"], [("dec_cut20", "REPLACES", "dec_cut16"), ("{who}", "DECIDES", "dec_cut20")], key=True, tag="cut20")
    say(69, 15.42, "priya", "Fine by me, it gives QA two more days on the token refresh path.", "auth", ["token_refresh"], [], support=True)
    say(69, 15.5, "tomas", "20 March works. The upgrade finishes on the 18th.", "auth", ["staging_cluster"], [], support=True)
    say(69, 15.8, "jonah", "Notice redrafted for 20 March. Goes out Monday.", "auth", ["tenant_notice"], [("tenant_notice", "ANNOUNCES", "dec_cut20")], support=True, tag="notice")
    # 12 March: the PR
    say(70, 10.25, "priya", "PR for the auth refactor is up. auth_v2 off by default, token refresh path rewritten, sessions untouched in Redis.", "auth",
        ["auth_v2", "token_refresh", "redis"], [("{who}", "SHIPS", "auth_v2")], key=True, tag="pr")
    say(70, 11.0, "mei", "Frontend side is ready, nothing to change on our end for auth_v2. Smoke test added.", "auth", ["auth_v2"], [], support=True)
    say(70, 11.33, "tomas", "Reviewed the infra bits of the auth PR. Approved.", "auth", ["auth_v2"], [], support=True)
    say(70, 16.67, "jonah", "Tenant notice for the 20 March cutover is approved by support. Goes out Monday.", "auth", ["tenant_notice"], [], support=True)
    say(70, 17.08, "dana", "Good. Retro item: we should have had the staging calendar in the design review.", "auth", ["retro", "design_review"], [], support=True)
    # afterwards
    say(75, 9.2, "mei", "March retro is on Thursday, agenda in the doc", "chatter", ["retro"], [])
    say(78, 9.67, "priya", "auth_v2 cutover done at 09:40, sessions intact, refresh errors zero.", "auth", ["auth_v2"], [("auth_v2", "CUT_OVER_ON", "dec_cut20")], tag="cutover")
    say(78, 9.75, "dana", "Nice work everyone.", "auth", [], [])
    say(103, 13.5, "jonah", "Remember when we thought the auth refactor would take a week.", "chatter", ["auth_service"], [])
    say(103, 13.55, "priya", "It did take a week. The other five were meetings.", "chatter", [], [])


for d in range(C):
    filler(d)
for topic, days, templates, (lo, hi) in EPISODES:
    for d in days:
        episode(d, templates, rng.randint(lo, hi), topic)
for d, items in INCIDENTS.items():
    for who, text, ents, rels in items:
        say(d, h(), who, text, "incident", ents, rels)
    for _ in range(rng.randint(18, 26)):
        say(d, h(), rng.choice(PEOPLE), rng.choice(["any update?", "watching the graphs", "status page updated", "I am around if needed", "customers are asking", "looks stable now"]), "incident", [], [])
search_decisions()
auth_thread()

MSGS.sort(key=lambda m: m["d"])
for i, m in enumerate(MSGS):
    m["id"] = f"m{i + 1:04d}"
by_tag = {m["tag"]: m for m in MSGS if m.get("tag")}

# ---------------------------------------------------------------- retrieval replay
STOP = set("what did we about the in a an of to is it and or for on at".split())
STEM = {"agreed": "agree", "agrees": "agree", "agreement": "agree", "refactoring": "refactor", "refactored": "refactor"}


def toks(s):
    return [STEM.get(t, t) for t in re.findall(r"[a-z0-9]+", s.lower())]


docs = [toks(m["text"]) for m in MSGS]
N = len(docs)
avgdl = sum(map(len, docs)) / N
df = {}
for t in docs:
    for w in set(t):
        df[w] = df.get(w, 0) + 1
qterms = [t for t in toks(QUERY) if t not in STOP]


def bm25(t, k1=1.5, b=0.75):
    s = 0.0
    for q in qterms:
        f = t.count(q)
        if not f:
            continue
        idf = math.log(1 + (N - df[q] + 0.5) / (df[q] + 0.5))
        s += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * len(t) / avgdl))
    return s


scored = sorted(((bm25(t), i) for i, t in enumerate(docs)), reverse=True)
K = 14
seeds = [dict(id=MSGS[i]["id"], score=round(s, 3), rank=r + 1) for r, (s, i) in enumerate(scored[:K]) if s > 0]


def dbscan(points, eps=1.5, min_pts=3):
    labels = [-1] * len(points)
    cid = 0
    for i in range(len(points)):
        if labels[i] != -1:
            continue
        nb = [j for j in range(len(points)) if abs(points[j] - points[i]) <= eps]
        if len(nb) < min_pts:
            continue
        labels[i] = cid
        queue = list(nb)
        while queue:
            j = queue.pop()
            if labels[j] == -1:
                labels[j] = cid
                nb2 = [k for k in range(len(points)) if abs(points[k] - points[j]) <= eps]
                if len(nb2) >= min_pts:
                    queue.extend(k for k in nb2 if labels[k] == -1)
        cid += 1
    return labels


seed_msgs = [next(m for m in MSGS if m["id"] == s["id"]) for s in seeds]
labels = dbscan([m["d"] for m in seed_msgs])
clusters = {}
for lab, m in zip(labels, seed_msgs):
    if lab >= 0:
        clusters.setdefault(lab, []).append(m)
best = max(clusters.values(), key=len)
w0, w1 = math.floor(min(m["d"] for m in best)), math.floor(max(m["d"] for m in best))
window = [m for m in MSGS if w0 <= math.floor(m["d"]) <= w1]
label_of = {m["id"]: ("key" if m["key"] else "support" if m["support"] else "irrelevant") for m in window}
links = [dict(type="update", frm=by_tag["cut20"]["id"], to=by_tag["outcome"]["id"], note="the cutover date was changed")]
for a, b in [("doc", "outcome"), ("objection", "outcome"), ("sched", "outcome"), ("mei_q", "flag"),
             ("staging_block", "cut20"), ("notice", "cut20"), ("pr", "flag")]:
    links.append(dict(type="expand", frm=by_tag[a]["id"], to=by_tag[b]["id"]))
kept = [m for m in window if label_of[m["id"]] != "irrelevant"]
forest_tokens = int(sum(len(toks(m["text"])) * 1.3 + 12 for m in kept))
answer = ("Keep Redis for sessions and rewrite only the token refresh path, behind the auth_v2 flag, off by default, "
          "with rollback by flipping it off. The cutover was set for 16 March at the design review on 10 March, "
          "then moved to 20 March on 11 March because the staging cluster had its Postgres upgrade that week. "
          "The PR went up on 12 March.")
citations = [by_tag[t]["id"] for t in ("outcome", "flag", "cut20", "pr")]

# ---------------------------------------------------------------- output
days = []
for d in range(C):
    dt = date(d)
    days.append(dict(index=d, date=dt.isoformat(), label=dlabel(d),
                     month=dt.strftime("%b") if dt.day == 1 else None, monday=dt.weekday() == 0))
first_d = {}
for m in MSGS:
    for e in m["ents"]:
        first_d.setdefault(e, m["d"])
entities = []
for e in E.values():
    if e["id"] not in first_d:
        continue
    entities.append(dict(id=e["id"], name=e["name"], label=e["label"], first_d=first_d[e["id"]],
                         first_appearance=int(first_d[e["id"]]), appearances=sorted(APPS[e["id"]]),
                         total_appearances=len(APPS[e["id"]]), properties=dict(description=e["description"])))
relations = [dict(source_id=s, target_id=t, label=l, first_d=r["first"], first_appearance=int(r["first"]),
                  appearances=sorted(r["apps"]), total_appearances=len(r["apps"]))
             for (s, l, t), r in sorted(RELS.items(), key=lambda kv: kv[1]["first"])]
messages = [dict(id=m["id"], d=m["d"], who=m["who"], text=m["text"], topic=m["topic"], ents=m["ents"]) for m in MSGS]
per_day = [0] * C
for m in MSGS:
    per_day[int(m["d"])] += 1
busiest = max(range(C), key=lambda d: per_day[d])

data = dict(
    metadata=dict(work_key="memory", work_name="A team's four months", series_no=4,
                  chapter_label="Day", total_days=C, total_messages=len(MSGS), total_entities=len(entities),
                  total_relations=len(relations), demo=True,
                  demo_note="Demo conversation, synthetic and narratively anchored",
                  period=f"{dlabel(0)} to {dlabel(C - 1)} 2026", busiest_day=dict(index=busiest, label=dlabel(busiest), count=per_day[busiest]),
                  replay="T-Mem, arXiv 2610.12874"),
    days=days, entities=entities, relations=relations, messages=messages,
    query=dict(text=QUERY, d=C, terms=qterms, seeds=seeds,
               episode=dict(start=w0, end=w1, start_label=dlabel(w0), end_label=dlabel(w1),
                            start_d=w0, end_d=w1 + 1, messages=len(window),
                            seeds_inside=sum(1 for m in best), seeds_total=len(seeds)),
               window=[m["id"] for m in window], labels=label_of,
               counts=dict(key=sum(1 for v in label_of.values() if v == "key"),
                           support=sum(1 for v in label_of.values() if v == "support"),
                           irrelevant=sum(1 for v in label_of.values() if v == "irrelevant")),
               links=links, tokens=dict(forest=forest_tokens, budget=BUDGET),
               answer=dict(text=answer, citations=citations),
               cost=dict(write_time_calls=len(MSGS), write_time_merges=sum(len(r["apps"]) - 1 for r in RELS.values()),
                         query_time_calls=1, query_time_messages=len(window))),
)

(HERE / "memory_kg.json").write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
tpl = (HERE / "template.html").read_text(encoding="utf-8")
blob = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
(HERE / "docs").mkdir(exist_ok=True)
(HERE / "docs" / "index.html").write_text(tpl.replace("__DATA__", blob), encoding="utf-8")

print(f"messages {len(MSGS)}, entities {len(entities)}, relations {len(relations)}, busiest {dlabel(busiest)} ({per_day[busiest]})")
print(f"query terms {qterms}")
print("seeds:")
for s, m in zip(seeds, seed_msgs):
    inside = "  in window" if w0 <= math.floor(m["d"]) <= w1 else ""
    print(f"  #{s['rank']:2d} {s['score']:.2f} {dlabel(m['d']):>6} {m['who']:6} {m['text'][:70]}{inside}")
print(f"episode {dlabel(w0)} to {dlabel(w1)}: {len(window)} messages, labels {data['query']['counts']}, forest {forest_tokens} tokens")
