"""Render results/jev_quantization_trial.html from the template and page data."""
import json
import re

data = json.load(open("results/page_data.json"))
for b in data["bars"]:  # shrink MusicXML: drop indentation between tags
    b["truth_xml"] = re.sub(r">\s+<", "><", b["truth_xml"])
    for s in b["settings"].values():
        for p in s["picks"].values():
            if "xml" in p:
                p["xml"] = re.sub(r">\s+<", "><", p["xml"])
blob = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
html = open("scripts/report_template.html").read().replace("/*__DATA__*/", blob)
open("results/jev_quantization_trial.html", "w").write(html)
print(len(html) // 1024, "KB")
