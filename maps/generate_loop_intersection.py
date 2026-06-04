#!/usr/bin/env python3
"""Build maps/loop_intersection.xodr by extending the known-good
simple_intersection.xodr (which CARLA parses cleanly).

Each arm gets:
  * extended to ARM_LENGTH m (outer end moved, junction end fixed)
  * 1 inbound driving lane (right id=-1) + sidewalk (right id=-2)
  * 1 outbound driving lane (left  id= 1) + sidewalk (left  id= 2)
  * broken yellow road-mark on the center lane (Town01-style)
  * sidewalk that tapers to 0 over the last TAPER_LEN m before the junction
  * a traffic-light signal on the inbound approach, scoped to lane -1
  * two controller elements grouping NS and EW signals into phases

Run:  python maps/generate_loop_intersection.py
Validate offline:
  LD_LIBRARY_PATH=.compatlibs .venv/bin/python -c "
      import sys, carla
      sys.path.insert(0,'carla/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg')
      m = carla.Map('t', open('maps/loop_intersection.xodr').read())
      print('OK', len(m.generate_waypoints(1.0)))"
"""
import math
import os
import xml.etree.ElementTree as ET

HERE = os.path.dirname(__file__)
SRC  = os.path.join(HERE, "simple_intersection.xodr")
OUT  = os.path.join(HERE, "loop_intersection.xodr")

ARM_LENGTH  = 250.0   # metres — long enough to observe realistic queue build-up
LANE_W      = 3.5
SIDEWALK_W  = 2.0
TAPER_LEN   = 8.0    # sidewalk tapers down over last N m before junction
# Taper to a small positive residual, NOT exactly 0: a lane whose width hits
# 0.0 makes CARLA's strict world.get_map() parser assert "distance > 0.0".
MIN_SIDEWALK_W = 0.1
# A lane on the inside of an arc whose width exceeds the arc radius folds past
# the centre of curvature (inner edge radius < 0) — same "distance > 0.0"
# assertion. Keep every connector lane this far under its arc radius.
CONNECTOR_RADIUS_MARGIN = 0.75

# Signal IDs keyed by arm road id (matches r.get("id") + "01")
# road_west=0→"001", road_east=1→"101", road_south=2→"201", road_north=3→"301"
_NS_SIG_IDS = ("301", "201")   # north + south
_EW_SIG_IDS = ("101", "001")   # east  + west


# ── XML helpers ───────────────────────────────────────────────────────────────

def _width(a, b=0.0):
    e = ET.Element("width")
    for k, v in (("sOffset", "0.0"), ("a", str(a)), ("b", str(b)),
                 ("c", "0.0"), ("d", "0.0")):
        e.set(k, v)
    return e


def _road_mark(typ="broken", color="yellow"):
    e = ET.Element("roadMark")
    for k, v in (("sOffset", "0.0"), ("type", typ), ("material", "standard"),
                 ("color", color), ("width", "0.125"), ("laneChange", "none")):
        e.set(k, v)
    return e


def _none_mark():
    return _road_mark(typ="none", color="white")


def _lane(lid, typ, w, b=0.0):
    e = ET.Element("lane")
    e.set("id", str(lid))
    e.set("type", typ)
    e.set("level", "false")
    e.append(_width(w, b))
    return e


def _arm_lane_section(s_val, sidewalk_b=0.0, with_center_mark=True):
    ls = ET.Element("laneSection")
    ls.set("s", str(s_val))

    left = ET.SubElement(ls, "left")
    l1 = _lane(1, "driving", LANE_W);           l1.append(_none_mark()); left.append(l1)
    l2 = _lane(2, "sidewalk", SIDEWALK_W, sidewalk_b); l2.append(_none_mark()); left.append(l2)

    center = ET.SubElement(ls, "center")
    cl = ET.SubElement(center, "lane")
    cl.set("id", "0"); cl.set("type", "none"); cl.set("level", "false")
    cl.append(_road_mark("broken") if with_center_mark else _none_mark())

    right = ET.SubElement(ls, "right")
    lm1 = _lane(-1, "driving", LANE_W);          lm1.append(_none_mark()); right.append(lm1)
    lm2 = _lane(-2, "sidewalk", SIDEWALK_W, sidewalk_b); lm2.append(_none_mark()); right.append(lm2)

    return ls


# ── geometry extension ────────────────────────────────────────────────────────

def _extend_arm(r, new_length):
    """Stretch arm road to new_length, keeping the junction-end fixed in world space."""
    geom = r.find("planView/geometry")
    old_x   = float(geom.get("x"))
    old_y   = float(geom.get("y"))
    old_len = float(geom.get("length"))
    hdg     = float(geom.get("hdg"))

    # World position of the junction end (= old road end)
    junc_x = old_x + old_len * math.cos(hdg)
    junc_y = old_y + old_len * math.sin(hdg)

    # New outer-end position (junction end fixed, outer end moves farther away)
    new_x = junc_x - new_length * math.cos(hdg)
    new_y = junc_y - new_length * math.sin(hdg)

    geom.set("x",      f"{new_x:.6f}")
    geom.set("y",      f"{new_y:.6f}")
    geom.set("length", f"{new_length:.6f}")
    r.set("length",    f"{new_length:.6f}")


def _clamp_connector_widths(r):
    """Shrink connector lane widths that exceed their arc radius.

    A lane on the inside of an arc whose constant width `a` is >= the arc
    radius produces an inner edge at radius <= 0 — degenerate geometry that
    trips CARLA's get_map() "distance > 0.0" assertion. We cap each lane's
    width at |R| - CONNECTOR_RADIUS_MARGIN. Straight connectors are untouched.
    """
    geom = r.find("planView/geometry")
    if geom is None:
        return
    arc = geom.find("arc")
    if arc is None:
        return
    curv = float(arc.get("curvature"))
    if curv == 0.0:
        return
    max_w = abs(1.0 / curv) - CONNECTOR_RADIUS_MARGIN
    if max_w <= 0.0:
        return
    for w in r.iter("width"):
        if float(w.get("a")) > max_w:
            w.set("a", f"{max_w:.4f}")


# ── main build ────────────────────────────────────────────────────────────────

def build():
    tree = ET.parse(SRC)
    root = tree.getroot()

    for r in root.findall("road"):
        if r.get("junction") != "-1":
            # Junction connector: clamp lane widths so a tight turn arc can't
            # fold its inside lane (inner edge radius must stay > 0).
            _clamp_connector_widths(r)
            continue

        # 1. Extend geometry, keeping junction end in place
        _extend_arm(r, ARM_LENGTH)
        length = ARM_LENGTH

        # 2. Rebuild lane sections
        taper_s = length - TAPER_LEN
        taper_b = -(SIDEWALK_W - MIN_SIDEWALK_W) / TAPER_LEN

        lanes_el = r.find("lanes")
        for ls in list(lanes_el.findall("laneSection")):
            lanes_el.remove(ls)
        lanes_el.append(_arm_lane_section(0.0,     sidewalk_b=0.0,    with_center_mark=True))
        lanes_el.append(_arm_lane_section(taper_s, sidewalk_b=taper_b, with_center_mark=False))

        # 3. Traffic-light signal
        #    s: stop-line position, 15 m from junction end (gives TM enough braking room)
        #    t: −(LANE_W + SIDEWALK_W/2) ≈ −4.5 m → pole at kerb face
        #    orientation: "+" = faces traffic traveling in +s direction (inbound)
        sig_id = r.get("id") + "01"
        sigs_el = ET.SubElement(r, "signals")
        sig = ET.SubElement(sigs_el, "signal")
        for k, v in {
            "s":           f"{length - 15:.2f}",
            "t":           f"{-(LANE_W + SIDEWALK_W / 2):.2f}",
            "id":          sig_id,
            "name":        "tl_" + r.get("name"),
            "dynamic":     "yes",
            "orientation": "+",
            "zOffset":     "5.0",
            "country":     "OpenDRIVE",
            "type":        "1000001",
            "subtype":     "-1",
            "value":       "-1",
            "height":      "3.0",
            "width":       "0.5",
        }.items():
            sig.set(k, v)
        # Scope signal to inbound lane only
        validity = ET.SubElement(sig, "validity")
        validity.set("fromLane", "-1")
        validity.set("toLane",   "-1")

    # 4. Phase controllers — group NS and EW signals so CARLA/TM knows the phases
    for ctrl_id, ctrl_name, sig_ids in (
        ("1", "ctrl_ns", _NS_SIG_IDS),
        ("2", "ctrl_ew", _EW_SIG_IDS),
    ):
        ctrl = ET.SubElement(root, "controller")
        ctrl.set("id", ctrl_id)
        ctrl.set("name", ctrl_name)
        ctrl.set("sequence", "0")
        for sid in sig_ids:
            c = ET.SubElement(ctrl, "control")
            c.set("signalId", sid)
            c.set("type", "")

    tree.write(OUT, encoding="UTF-8", xml_declaration=True)
    return OUT


if __name__ == "__main__":
    print("wrote", build())
