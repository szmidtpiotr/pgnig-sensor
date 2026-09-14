"""The marginal price fit and the entity that publishes it."""
from datetime import UTC, datetime, timedelta

import pytest
from homeassistant.components.sensor import SensorDeviceClass
from homeassistant.core import HomeAssistant

from custom_components.pgnig_gas_sensor.sensor import (
    PgnigMarginalPriceSensor,
    PgnigReadingDateSensor,
    _billing_days,
    _fit_marginal_price,
)

from .builders import build_stub_coordinator, make_invoice, make_reading


def invoice_with(volume, amount, day=1, days=None):
    """Invoice for meter id_pp="1" covering `volume` m³ billed at `amount` gross.

    `days` sets the billing period length; None strips the dates so the
    per-invoice fallback is exercised.
    """
    start = datetime(2026, 1, 1)
    return make_invoice(
        wear_m3=volume,
        wear=volume,
        gross_amount=amount,
        date=datetime(2026, 1, day),
        start_date=start if days else None,
        end_date=(start + timedelta(days=days)) if days else None,
    )


# --- the fit -----------------------------------------------------------


def test_marginal_price_separates_fixed_from_variable():
    """amount = fixed + marginal * volume, recovered exactly from clean points."""
    # 40 PLN standing charge, 3 PLN per m³.
    invoices = [invoice_with(33, 40 + 3 * 33), invoice_with(300, 40 + 3 * 300)]
    marginal, fixed, used, model, _ = _fit_marginal_price(invoices)
    assert model == "per_invoice"
    assert marginal == pytest.approx(3.0)
    assert fixed == pytest.approx(40.0)
    assert used == 2


def test_marginal_price_needs_two_distinct_volumes():
    """One invoice, or several at the same volume, leaves the split undetermined."""
    assert _fit_marginal_price([]) is None
    assert _fit_marginal_price([invoice_with(33, 139)]) is None
    assert _fit_marginal_price([invoice_with(33, 139), invoice_with(33, 139)]) is None


def test_marginal_price_rejects_non_positive_rate():
    """A falling amount against a rising volume is not a tariff."""
    assert _fit_marginal_price([invoice_with(33, 300), invoice_with(300, 100)]) is None


def test_billing_days_reads_the_period():
    assert _billing_days(invoice_with(33, 100, days=61)) == 61
    assert _billing_days(invoice_with(33, 100)) is None


def test_fit_uses_days_when_dates_are_present():
    """Standing charges accrue per day, so a longer period carries more of them.

    2 PLN/day plus 3 PLN/m³: a 30-day and a 60-day invoice at the same volume
    differ only by the standing charge, which the per-invoice model cannot
    separate at all.
    """
    invoices = [
        invoice_with(33, 2 * 30 + 3 * 33, days=30),
        invoice_with(300, 2 * 60 + 3 * 300, days=60),
    ]
    marginal, per_day, used, model, _ = _fit_marginal_price(invoices)
    assert model == "per_day"
    assert marginal == pytest.approx(3.0)
    assert per_day == pytest.approx(2.0)
    assert used == 2


def test_fit_falls_back_when_dates_are_missing():
    """Without usable dates the per-invoice split still applies."""
    invoices = [invoice_with(33, 40 + 3 * 33), invoice_with(300, 40 + 3 * 300)]
    marginal, fixed, _, model, _ = _fit_marginal_price(invoices)
    assert model == "per_invoice"
    assert marginal == pytest.approx(3.0)
    assert fixed == pytest.approx(40.0)


def test_equal_periods_still_recover_the_rate():
    """Equal periods are not degenerate; the per-day charge is just fixed/days.

    Volume still varies, so the slope is determined. The standing charge comes
    out as a rate rather than a lump sum, which is the same information.
    """
    invoices = [
        invoice_with(33, 40 + 3 * 33, days=30),
        invoice_with(300, 40 + 3 * 300, days=30),
    ]
    marginal, per_day, _, model, _ = _fit_marginal_price(invoices)
    assert model == "per_day"
    assert marginal == pytest.approx(3.0)
    assert per_day * 30 == pytest.approx(40.0)


def test_fit_uses_only_the_most_recent_invoices():
    """An old tariff outside the window must not drag the current rate."""
    # Six invoices at 40 PLN + 3 PLN/m³ fill the window, newest first.
    recent = [
        invoice_with(volume, 40 + 3 * volume, day=20 + index)
        for index, volume in enumerate([10, 50, 100, 150, 200, 300])
    ]
    # Older ones priced at a different tariff sit outside it.
    stale = [
        invoice_with(volume, 500 + 0.1 * volume, day=1 + index)
        for index, volume in enumerate([20, 60, 120, 180])
    ]
    marginal, fixed, used, _, _ = _fit_marginal_price(stale + recent)
    assert used == 6
    assert marginal == pytest.approx(3.0)
    assert fixed == pytest.approx(40.0)


# --- the entity --------------------------------------------------------


async def test_marginal_price_entity_reports_the_fitted_rate(hass: HomeAssistant):
    """State is the rate alone; the standing charge lands in the attributes."""
    coordinator = build_stub_coordinator(
        hass,
        invoices=[
            invoice_with(33, 40 + 3 * 33, day=1),
            invoice_with(300, 40 + 3 * 300, day=2),
        ],
    )
    sensor = PgnigMarginalPriceSensor(coordinator, "M1", 1)

    assert sensor.state == pytest.approx(3.0)
    attrs = sensor.extra_state_attributes
    assert attrs["model"] == "per_invoice"
    assert attrs["fixed_charge_per_invoice"] == pytest.approx(40.0)
    assert attrs["invoices_used"] == 2
    assert attrs["residual_pct"] == pytest.approx(0.0, abs=1e-6)


async def test_marginal_price_entity_reports_per_day_charge(hass: HomeAssistant):
    """The per-day model also converts the charge for the newest invoice."""
    coordinator = build_stub_coordinator(
        hass,
        invoices=[
            invoice_with(33, 2 * 30 + 3 * 33, day=1, days=30),
            invoice_with(300, 2 * 60 + 3 * 300, day=2, days=60),
        ],
    )
    sensor = PgnigMarginalPriceSensor(coordinator, "M1", 1)

    attrs = sensor.extra_state_attributes
    assert attrs["model"] == "per_day"
    assert attrs["fixed_charge_per_day"] == pytest.approx(2.0)
    assert attrs["fixed_charge_per_invoice"] == pytest.approx(120.0)


async def test_marginal_price_entity_silent_when_undetermined(hass: HomeAssistant):
    """A single invoice must not produce a made-up rate."""
    coordinator = build_stub_coordinator(hass, invoices=[invoice_with(33, 139)])
    sensor = PgnigMarginalPriceSensor(coordinator, "M1", 1)

    assert sensor.state is None
    assert sensor.extra_state_attributes == {}


async def test_marginal_price_entity_ignores_other_meters(hass: HomeAssistant):
    """Invoices belonging to a second meter must not enter this meter's fit."""
    coordinator = build_stub_coordinator(
        hass,
        invoices=[
            invoice_with(33, 40 + 3 * 33, day=1),
            invoice_with(300, 40 + 3 * 300, day=2),
            make_invoice(id_pp="2", wear_m3=500, wear=500, gross_amount=5.0),
        ],
    )
    sensor = PgnigMarginalPriceSensor(coordinator, "M1", 1)

    assert sensor.extra_state_attributes["invoices_used"] == 2
    assert sensor.state == pytest.approx(3.0)


async def test_marginal_price_entity_without_a_poll(hass: HomeAssistant):
    """Before the first successful poll the entity reports nothing."""
    coordinator = build_stub_coordinator(hass, has_data=False)
    sensor = PgnigMarginalPriceSensor(coordinator, "M1", 1)

    assert sensor.state is None
    assert sensor.extra_state_attributes == {}


# --- the reading date entity -------------------------------------------


async def test_reading_date_entity_is_an_aware_timestamp(hass: HomeAssistant):
    """EBOK sends ReadingDateUtc naive; HA rejects naive datetimes here."""
    reading = make_reading(reading_date_utc=datetime(2026, 3, 4, 5, 6))
    coordinator = build_stub_coordinator(hass, readings={"M1": reading})
    sensor = PgnigReadingDateSensor(coordinator, "M1", 1)

    assert sensor.device_class == SensorDeviceClass.TIMESTAMP
    assert sensor.state == datetime(2026, 3, 4, 5, 6, tzinfo=UTC)
    attrs = sensor.extra_state_attributes
    assert attrs["reading_type"] == reading.type
    assert attrs["reading_status"] == reading.status
    assert attrs["value"] == reading.value


async def test_reading_date_entity_without_a_reading(hass: HomeAssistant):
    coordinator = build_stub_coordinator(hass, readings={})
    sensor = PgnigReadingDateSensor(coordinator, "M1", 1)

    assert sensor.state is None
    assert sensor.extra_state_attributes == {}
