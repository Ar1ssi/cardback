"""The editor page's project sync and ptcg-sim send queue (gui.html), read from the page source. The behaviour tests
run the page's own functions in Node (skipped without it); no browser, network or server.
Run: python -m unittest discover tests"""
import json
import re
import shutil
import subprocess
import unittest
from pathlib import Path

from svrestyle import gui, projects

PAGE = (Path(gui.__file__).with_name("gui.html")).read_text(encoding="utf-8")
SCRIPT = PAGE[PAGE.index('<script>\n"use strict";'):]


def function(name):
    """The source of a top-level `function name(...) {...}` of the page script (it closes at a '}' in column 0)."""
    m = re.search(rf"^(?:async )?function {name}\(.*?^}}$", SCRIPT, re.S | re.M)
    if not m:
        raise AssertionError(f"no function {name} in gui.html")
    return m.group(0)


def const(name):
    """The source of a top-level `const name = ...;` that fits on one line."""
    m = re.search(rf"^const {name} = .*;$", SCRIPT, re.M)
    if not m:
        raise AssertionError(f"no const {name} in gui.html")
    return m.group(0)


def node(program):
    out = subprocess.run(["node", "-e", program], capture_output=True, text=True, timeout=30)
    if out.returncode:
        raise AssertionError(out.stderr)
    return json.loads(out.stdout)


class PageState(unittest.TestCase):
    def test_state_names_each_field_once(self):
        block = re.search(r"^const state = \{\n(.*?)^\};", SCRIPT, re.S | re.M).group(1)
        names = re.findall(r"^  (\w+):", block, re.M)
        self.assertEqual(sorted({n for n in names if names.count(n) > 1}), [],
                         "a field declared twice is one field: two queues would share it")


@unittest.skipUnless(shutil.which("node"), "needs Node")
class PageSync(unittest.TestCase):
    def test_every_project_field_is_patched_and_the_server_reads_it_the_same(self):
        sync = "\n".join([const("sameJSON"), const("emptyData"), const("withDefaults"), const("PATCH_FIELDS"),
                           function("diffData"), function("applyPatch"), function("reorder")])
        out = node(sync + """
          const local = { ...emptyData(), setName: "S", foil: true, current: "a", code: "HBW" };
          const patch = diffData(local, emptyData());
          const fields = Object.keys(emptyData()).filter((k) => !["version", "cards", "edits", "options"].includes(k));
          console.log(JSON.stringify({ patch, fields, PATCH_FIELDS, applied: applyPatch(emptyData(), patch) }));
        """)
        self.assertEqual(sorted(out["fields"]), sorted(out["PATCH_FIELDS"]), "projectData's fields, all patched")
        self.assertEqual(out["patch"], {"setName": "S", "foil": True, "current": "a", "code": "HBW"})
        self.assertEqual(out["applied"]["code"], "HBW")
        server = projects.apply_patch({"version": 1, "cards": []}, out["patch"])
        self.assertEqual({k: server[k] for k in out["PATCH_FIELDS"]}, {k: out["applied"][k] for k in out["PATCH_FIELDS"]})


@unittest.skipUnless(shutil.which("node"), "needs Node")
class SendQueue(unittest.TestCase):
    """A send asked for while one runs keeps what it was asked to send; the project's save queue is apart."""

    def run_queue(self, steps):
        return node("\n".join([
            "const state = { pushing: true, publishing: false, sendQueue: [], saveAgain: false };",
            "const sent = [];",
            # stand-ins: each starts at once, which is when the page's own set state.pushing / state.publishing
            "function pushSimSet(o) { if (state.pushing || state.publishing) return queueSend('push', o); "
            "  state.pushing = true; sent.push(['push', o]); }",
            "function publishSiteSet(o) { if (state.pushing || state.publishing) return queueSend('publish', o); "
            "  state.publishing = true; sent.push(['publish', o]); }",
            "const finish = () => { state.pushing = state.publishing = false; runQueuedPush(); };",
            function("queueSend"), function("runQueuedPush"), steps,
            "console.log(JSON.stringify({ sent, saveAgain: state.saveAgain }));"]))

    def test_a_queued_push_all_sends_its_own_list_and_force_sticks(self):
        out = self.run_queue("""
          const plan = { items: [{ id: "p" }], title: "All", code: "ALL" };
          pushSimSet({ set: plan });
          pushSimSet({ force: true, set: null });
          pushSimSet({ set: null });
          publishSiteSet({ set: plan, force: false, allowShrink: false });
          state.saveAgain = true;                       // a project save waiting meanwhile
          finish(); finish(); finish();
        """)
        self.assertEqual([(kind, o.get("set") and o["set"]["code"], o.get("force")) for kind, o in out["sent"]],
                         [("push", "ALL", None), ("push", None, True), ("publish", "ALL", False)])
        self.assertTrue(all(o["quiet"] for kind, o in out["sent"] if kind == "push"), "queued sends never ask")
        self.assertTrue(out["saveAgain"], "running the send queue leaves the save queue alone")


if __name__ == "__main__":
    unittest.main()
