#!/usr/bin/env python3
from pathlib import Path

from jinja2 import Environment, PackageLoader


def generate(idx, policy_links, outdir):
    env = Environment(
        loader=PackageLoader("vot_report_generator", "templates"))
    template = env.get_template("index.html")
    outdir.mkdir(exist_ok=True, parents=True)
    with open(outdir / "index.html", 'w') as f:
        f.write(template.render(
            idx=idx,
            policy_links=policy_links))
