# 20261117T061915-1716ccff

Status: **blocked**

User selection (verbatim): 피자로 진행해볼까?
User priority (verbatim): 최대한 구현하기 쉬우면서도 기존과 핵심 조작 능력이 겹치지 않는 것들 위주로 생각해봐

The selected proposal is to turn a partially baked pizza in place with a small turning peel inside a real-scale oven. Retain this direction. The underside has set enough to handle. Starting on the stone, bring the pale rim around toward the hot side while keeping the pizza near its initial center, top side up, supported by the deck, and inside the oven throughout. Withdraw the peel. Proposed target: roughly half a turn, with visible browning asymmetry. Assess the physics of manipulating the food, not cooking chemistry or food safety. Score true yaw change, center excursion throughout the trajectory, tool and deck support/contact, and final release. Negative controls include bypass by gripper, translating the pizza out to rotate, and loss of stone support. Use YAM's actual gripper controls and existing common task structure. Candidate choices should stay within this same skill. Use a physically justified minimal crust contact model, including rim compliance if it materially affects insertion. Do not weld the tool to the pizza or drive pizza pose.
First test full-scale reach and clearance of the handle, oven mouth and peel tip with the real YAM model, and blade insertion and tangential traction with no scripted food pose. Use existing submission-sheet comparisons, then focus on close cases. Official rejected launch PR #720 (https://github.com/robocurve/benchmark-tasks/pull/720#issuecomment-6816792297) had a 90 mm pizza, open-topped oven and rigid raw dough. Use a representative closed-roof pizza oven, a real turning peel and a roughly 30 cm pizza. Do not reduce the equipment scale to fix a reach problem. Sources: https://eu.ooni.com/products/ooni-koda-16 and https://ooni.com/products/ooni-pizza-turning-peel and https://ooni.com/blogs/ooni-insights/how-to-pick-the-right-pizza-peel . Distinguish actual measurements from provisional settings. Signed human review is for this new task separately before paid model inference or official submission.


incomplete evidence reference: source_id='s-6ff9ae104970e4bc', known_source=True, has_locator=True, quote_chars=6 (minimum 12). Use an exact acquired-text excerpt; inspect images separately, rather than quoting a short figure label.

- Release: `release/`
- Evidence: `workspace/artifacts/`
- This report is not submission approval.
