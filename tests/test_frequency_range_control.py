from math import log10

from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QSignalSpy, QTest

from blab.ui.frequency_range_control import FrequencyRangeControl


def enter_hz(spin, value):
    spin.setFocus()
    spin.selectAll()
    QTest.keyClicks(spin, str(value))
    QTest.keyClick(spin, Qt.Key_Return)


def test_typed_endpoints_clip_and_restore_silently(qapp):
    control = FrequencyRangeControl(1001, 5000)
    edits = QSignalSpy(control.edited)
    enter_hz(control.maximum_spin, 500)
    assert (control.minimum_spin.value(), control.maximum_spin.value()) == (1001, 1002)
    enter_hz(control.minimum_spin, 1500)
    assert control.minimum_spin.value() == 1001
    assert edits.count() == 1
    control.set_range(8, 40000)
    assert control.slider.value() == (log10(8), log10(40000))
    control.set_range(500, 500)
    assert control.minimum_spin.value() == control.maximum_spin.value() == 500
    assert edits.count() == 1


def test_drag_clamps_each_handle_and_commits_on_release(qapp):
    control = FrequencyRangeControl(1001, 5000)
    control.resize(700, 35)
    control.show()
    qapp.processEvents()
    slider = control.slider
    edits = QSignalSpy(control.edited)

    for start, end, expected in ((0.57, 1, (4999, 5000)), (0.82, 0, (1001, 1002))):
        control.set_range(1001, 5000)
        y = slider.height() // 2
        QTest.mousePress(slider, Qt.LeftButton, pos=QPoint(round(start * slider.width()), y))
        QTest.mouseMove(slider, QPoint(round(end * (slider.width() - 1)), y))
        assert (control.minimum_spin.value(), control.maximum_spin.value()) == expected
        before = edits.count()
        QTest.mouseRelease(slider, Qt.LeftButton, pos=QPoint(round(end * (slider.width() - 1)), y))
        assert edits.count() == before + 1
    assert edits.count() == 2
    control.close()


def test_overlapping_grips_hover_and_lock_through_reversal(qapp):
    control = FrequencyRangeControl(10000, 10001)
    control.resize(600, 35)
    control.show()
    qapp.processEvents()
    slider = control.slider
    for index in (0, 1):
        control.set_range(10000, 10001)
        low, high = slider._anchors()
        assert low == high
        point = slider._grip(index, low).center()
        QTest.mouseMove(slider, point)
        spin = (control.minimum_spin, control.maximum_spin)[index]
        assert spin.palette().base().color() == control.palette().highlight().color()
        QTest.mousePress(slider, Qt.LeftButton, pos=point)
        delta = QPoint(30 if index else -30, 0)
        QTest.mouseMove(slider, point + delta)
        assert (spin.value() > 10001) if index else (spin.value() < 10000)
        QTest.mouseMove(slider, point - delta)
        QTest.mouseRelease(slider, Qt.LeftButton, pos=point - delta)
        assert (control.minimum_spin.value(), control.maximum_spin.value()) == (10000, 10001)
    control.close()


def test_shared_seam_uses_first_deliberate_direction(qapp):
    control = FrequencyRangeControl(10000, 10001)
    control.resize(600, 35)
    control.show()
    qapp.processEvents()
    slider = control.slider
    for direction in (-1, 1):
        control.set_range(10000, 10001)
        seam = QPoint(slider._anchors()[0], slider.height() // 2)
        QTest.mousePress(slider, Qt.LeftButton, pos=seam)
        QTest.mouseMove(slider, seam + QPoint(direction, 0))
        assert (control.minimum_spin.value(), control.maximum_spin.value()) == (10000, 10001)
        end = seam + QPoint(direction * 30, 0)
        QTest.mouseMove(slider, end)
        QTest.mouseRelease(slider, Qt.LeftButton, pos=end)
        if direction < 0:
            assert control.minimum_spin.value() < 10000 and control.maximum_spin.value() == 10001
        else:
            assert control.minimum_spin.value() == 10000 and control.maximum_spin.value() > 10001
    control.close()
