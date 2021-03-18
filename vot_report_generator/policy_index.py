#!/usr/bin/env python3
from jinja2 import Environment, PackageLoader


def generate(name, link, fps, idx, test_results):
    env = Environment(
        loader=PackageLoader("vot_report_generator", "templates"))
    template = env.get_template("policy_index.html")
    index_path = 'output' / link / 'index.html'
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with open(index_path, 'w') as f:
        f.write(template.render(
            idx=idx,
            name=name,
            link=link,
            fps=fps,
            test_results=test_results))
