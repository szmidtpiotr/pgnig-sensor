"""Platform for sensor integration."""
from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any, Callable, Optional, Sequence

import homeassistant.helpers.config_validation as cv
import voluptuous as vol
from homeassistant.components.sensor import (
    PLATFORM_SCHEMA,
    SensorDeviceClass,
    SensorEntity,
    SensorStateClass,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_PASSWORD, CONF_USERNAME, UnitOfVolume
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .const import DEFAULT_AUTH_METHOD, DOMAIN
from .coordinator import PgnigCoordinator, PgnigData
from .Invoices import InvoicesList
from .PgnigApi import PgnigApi

_LOGGER = logging.getLogger(__name__)
PLATFORM_SCHEMA = PLATFORM_SCHEMA.extend({
    vol.Required(CONF_USERNAME): cv.string,
    vol.Required(CONF_PASSWORD): cv.string,
})

# Tariffs change. Fitting across every invoice the endpoint returns averages
# old and new prices into a rate that matches neither, so the fit uses only the
# most recent invoices. Six covers roughly a year of two-month periods -- enough
# points to be stable, recent enough to describe the tariff in force now.
MARGINAL_PRICE_INVOICE_WINDOW = 6


def _as_utc(value: datetime | None) -> datetime | None:
    """Return an aware UTC datetime; EBOK sends ReadingDateUtc without a suffix."""
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _billing_days(invoice: InvoicesList) -> int | None:
    """Length of the billing period in days, or None if the dates are unusable."""
    start = getattr(invoice, "start_date", None)
    end = getattr(invoice, "end_date", None)
    if start is None or end is None:
        return None
    days = (end - start).days
    return days if days > 0 else None


def _fit_marginal_price(
    invoices: Sequence[InvoicesList],
) -> tuple[float, float, int, str, float | None] | None:
    """Split invoices into a per-m3 rate and the standing charges.

    PgnigCostTrackingSensor reports gross_amount / volume, which folds the
    standing charges (subscription, fixed distribution fee) into the rate. That
    makes the figure rise as consumption falls -- the opposite of how a price
    behaves -- so it overstates cost when used as a price, worst outside the
    heating season.

    Two models, preferred in order:

    Per day, when the invoices carry usable start and end dates:

        amount = per_day * days + marginal * volume

    Standing charges accrue per day, not per invoice, so a two-month invoice
    carries twice the subscription of a one-month one. Fitting against days
    keeps periods of different length from distorting the split.

    Per invoice, as a fallback:

        amount = fixed + marginal * volume

    Returns (marginal, standing, invoices_used, model, residual_pct) where
    `standing` is PLN/day for the "per_day" model and PLN/invoice for
    "per_invoice". Returns None when the data cannot determine the split.
    """
    usable = [
        x for x in invoices
        if (x.wear_m3 or x.wear) and x.gross_amount is not None
    ]
    usable.sort(key=lambda x: x.date, reverse=True)
    usable = usable[:MARGINAL_PRICE_INVOICE_WINDOW]

    dated = [
        (_billing_days(x), x.wear_m3 or x.wear, x.gross_amount)
        for x in usable
    ]
    with_days = [(d, v, y) for d, v, y in dated if d is not None]

    if len(with_days) >= 2:
        fit = _solve_two_term(with_days)
        if fit is not None:
            per_day, marginal = fit
            return (
                marginal, per_day, len(with_days), "per_day",
                _mean_residual_pct(with_days, per_day, marginal),
            )

    points = [(v, y) for _, v, y in dated]
    fit = _solve_with_intercept(points)
    if fit is None:
        return None
    fixed, marginal = fit
    rows = [(1.0, v, y) for v, y in points]
    return (
        marginal, fixed, len(points), "per_invoice",
        _mean_residual_pct(rows, fixed, marginal),
    )


def _mean_residual_pct(
    rows: list[tuple[float, float, float]], per_day: float, marginal: float
) -> float | None:
    """Mean absolute error of the fit, as a percent of the mean invoice.

    Exposed so the fit does not have to be taken on trust: a few percent means
    the model describes the bills, a large figure means something the model does
    not capture -- a tariff change inside the window, a correction, a discount.
    """
    if not rows:
        return None
    total = sum(y for _, _, y in rows)
    if total == 0:
        return None
    error = sum(abs(per_day * d + marginal * v - y) for d, v, y in rows)
    return error / total * 100


def _solve_two_term(rows: list[tuple[float, float, float]]) -> tuple[float, float] | None:
    """Least squares for y = a*days + b*volume, no intercept."""
    sum_dd = sum(d * d for d, _, _ in rows)
    sum_dv = sum(d * v for d, v, _ in rows)
    sum_vv = sum(v * v for _, v, _ in rows)
    sum_dy = sum(d * y for d, _, y in rows)
    sum_vy = sum(v * y for _, v, y in rows)

    determinant = sum_dd * sum_vv - sum_dv * sum_dv
    if determinant == 0:
        # Days and volume move together across every invoice, so their effects
        # cannot be told apart.
        return None

    a = (sum_dy * sum_vv - sum_vy * sum_dv) / determinant
    b = (sum_vy * sum_dd - sum_dy * sum_dv) / determinant
    if b <= 0 or a < 0:
        return None
    return a, b


def _solve_with_intercept(points: list[tuple[float, float]]) -> tuple[float, float] | None:
    """Least squares for y = intercept + slope*x."""
    n = len(points)
    if n < 2:
        return None

    sum_x = sum(p[0] for p in points)
    sum_y = sum(p[1] for p in points)
    sum_xy = sum(p[0] * p[1] for p in points)
    sum_xx = sum(p[0] * p[0] for p in points)

    denominator = n * sum_xx - sum_x * sum_x
    if denominator == 0:
        # Every invoice covers the same volume; the split is undetermined.
        return None

    slope = (n * sum_xy - sum_x * sum_y) / denominator
    intercept = (sum_y - slope * sum_x) / n
    if slope <= 0:
        # A non-positive rate means the data does not describe a tariff, most
        # likely a corrected or re-issued invoice. Better to report nothing.
        return None
    return intercept, slope


def invoice_summary(
    invoices: Sequence[InvoicesList], id_local: int
) -> dict[str, Any]:
    """Unpaid total and the next payment due for one meter."""

    def upcoming_payment_for_meter(x: InvoicesList) -> bool:
        return str(id_local) == str(x.id_pp) and not x.is_paid and not x.is_credit_note

    unpaid_invoices = list(filter(upcoming_payment_for_meter, invoices))
    sum_of_unpaid_invoices = sum(x.amount_to_pay for x in unpaid_invoices)
    next_payment_item = (
        min(unpaid_invoices, key=lambda z: z.date) if unpaid_invoices else None
    )

    return {
        "sumOfUnpaidInvoices": sum_of_unpaid_invoices,
        "nextPaymentDate": next_payment_item.paying_deadline_date if next_payment_item else None,
        "nextPaymentWear": next_payment_item.wear_m3 or next_payment_item.wear if next_payment_item else None,
        "nextPaymentWearKWH": next_payment_item.wear_kwh if next_payment_item else None,
        "nextPaymentAmountToPay": next_payment_item.amount_to_pay if next_payment_item else None,
    }


def priced_invoices(
    invoices: Sequence[InvoicesList], id_local: int
) -> list[InvoicesList]:
    """Every invoice for one meter that carries a usable price per m³."""

    def has_valid_consumption(x: InvoicesList) -> bool:
        gas_m3 = x.wear_m3 or x.wear
        return (
            str(id_local) == str(x.id_pp)
            and gas_m3 is not None
            and gas_m3 != 0
            and x.gross_amount is not None
            and x.gross_amount != 0
            and not x.is_credit_note
        )

    return list(filter(has_valid_consumption, invoices))


def latest_priced_invoice(
    invoices: Sequence[InvoicesList], id_local: int
) -> InvoicesList | None:
    """The newest invoice for one meter that carries a usable price per m³."""
    valid_invoices = priced_invoices(invoices, id_local)
    return max(valid_invoices, key=lambda z: z.date) if valid_invoices else None


def entities_for_meters(coordinator: PgnigCoordinator) -> list[SensorEntity]:
    """Every sensor the given coordinator feeds."""
    return [
        entity
        for meter in coordinator.meters.ppg_list
        for entity in (
            PgnigSensor(
                coordinator, meter.meter_number, meter.id_local, tariff=meter.tariff
            ),
            PgnigInvoiceSensor(coordinator, meter.meter_number, meter.id_local),
            PgnigCostTrackingSensor(coordinator, meter.meter_number, meter.id_local),
            PgnigReadingDateSensor(coordinator, meter.meter_number, meter.id_local),
            PgnigMarginalPriceSensor(coordinator, meter.meter_number, meter.id_local),
        )
    ]


async def async_setup_entry(
        hass: HomeAssistant,
        config_entry: ConfigEntry,
        async_add_entities,
):
    runtime = hass.data[DOMAIN][config_entry.entry_id]
    async_add_entities(entities_for_meters(runtime.coordinator))


async def async_setup_platform(
        hass: HomeAssistant,
        config: ConfigType,
        async_add_entities: Callable,
        discovery_info: Optional[DiscoveryInfoType] = None,
) -> None:
    api = PgnigApi(config.get(CONF_USERNAME), config.get(CONF_PASSWORD), DEFAULT_AUTH_METHOD)
    meters = await hass.async_add_executor_job(api.meterList)
    coordinator = PgnigCoordinator(hass, None, api, meters)
    await coordinator.async_refresh()
    async_add_entities(entities_for_meters(coordinator))


class PgnigBaseSensor(CoordinatorEntity[PgnigCoordinator], SensorEntity):
    """Shared identity and state refresh for one meter's sensors.

    State is recomputed when the coordinator delivers a poll, so the entities
    never touch the API themselves.
    """

    _name_prefix: str
    _unique_id_prefix: str

    def __init__(
        self, coordinator: PgnigCoordinator, meter_id: str, id_local: int
    ) -> None:
        super().__init__(coordinator)
        self.meter_id = meter_id
        self.id_local = id_local
        self.entity_name = f"{self._name_prefix} {meter_id} {id_local}"
        self._state: Any = None
        self._refresh_state()

    @property
    def unique_id(self) -> str | None:
        return f"{self._unique_id_prefix}{self.meter_id}_{self.id_local}"

    @property
    def device_info(self):
        return {
            "identifiers": {(DOMAIN, self.meter_id)},
            "name": f"Orlen GAS METER ID {self.meter_id}",
            "manufacturer": "Orlen",
            "model": self.meter_id,
        }

    @property
    def name(self) -> str:
        return self.entity_name

    def _state_from(self, data: PgnigData) -> Any:
        """Derive this entity's state from one poll."""
        raise NotImplementedError

    def _refresh_state(self) -> None:
        data = self.coordinator.data
        self._state = None if data is None else self._state_from(data)

    @callback
    def _handle_coordinator_update(self) -> None:
        self._refresh_state()
        _LOGGER.debug("%s updated: %s", self.entity_name, self.state)
        super()._handle_coordinator_update()


class PgnigSensor(PgnigBaseSensor):
    """Latest meter reading."""

    _name_prefix = "Orlen Gas Sensor"
    _unique_id_prefix = "pgnig_sensor"

    _attr_native_unit_of_measurement = UnitOfVolume.CUBIC_METERS
    _attr_device_class = SensorDeviceClass.GAS
    _attr_state_class = SensorStateClass.TOTAL_INCREASING

    def __init__(
        self,
        coordinator: PgnigCoordinator,
        meter_id: str,
        id_local: int,
        tariff: str | None = None,
    ) -> None:
        self.tariff = tariff
        super().__init__(coordinator, meter_id, id_local)

    def _state_from(self, data: PgnigData):
        return data.readings.get(self.meter_id)

    @property
    def state(self):
        if self._state is None:
            return None
        return self._state.value

    @property
    def extra_state_attributes(self):
        attrs = dict()
        if self.tariff:
            attrs["tariff"] = self.tariff
        if self._state is not None:
            attrs["wear"] = self._state.wear
            attrs["wear_unit_of_measurment"] = UnitOfVolume.CUBIC_METERS
            attrs["reading_date"] = _as_utc(self._state.reading_date_utc)
            attrs["reading_date_local"] = self._state.reading_date_local
            attrs["reading_type"] = self._state.type
            attrs["reading_status"] = self._state.status
        return attrs


class PgnigInvoiceSensor(PgnigBaseSensor):
    """Total still owed, plus the next payment due."""

    _name_prefix = "Orlen Gas Invoice Sensor"
    _unique_id_prefix = "pgnig_invoice_sensor"

    _attr_native_unit_of_measurement = "PLN"
    _attr_device_class = SensorDeviceClass.MONETARY
    _attr_state_class = SensorStateClass.MEASUREMENT

    def _state_from(self, data: PgnigData):
        return invoice_summary(data.invoices, self.id_local)

    @property
    def state(self):
        if self._state is None:
            return None
        return self._state.get("sumOfUnpaidInvoices")

    @property
    def extra_state_attributes(self):
        attrs = dict()
        if self._state is not None:
            attrs["next_payment_date"] = self._state.get("nextPaymentDate")
            attrs["next_payment_amount_to_pay"] = self._state.get("nextPaymentAmountToPay")
            attrs["next_payment_wear"] = self._state.get("nextPaymentWear")
            attrs["next_payment_wear_KWH"] = self._state.get("nextPaymentWearKWH")
        return attrs


class PgnigCostTrackingSensor(PgnigBaseSensor):
    """Price per m³ from the most recent priced invoice."""

    _name_prefix = "Orlen Gas Cost Tracking Sensor"
    _unique_id_prefix = "pgnig_cost_tracking_sensor"

    _attr_native_unit_of_measurement = "PLN/m³"
    _attr_device_class = SensorDeviceClass.MONETARY
    _attr_state_class = SensorStateClass.MEASUREMENT

    def _state_from(self, data: PgnigData):
        return latest_priced_invoice(data.invoices, self.id_local)

    @property
    def state(self):
        if self._state is None:
            return None
        gas_m3 = self._state.wear_m3 or self._state.wear
        if self._state.gross_amount is None or gas_m3 is None or gas_m3 == 0:
            return None
        return self._state.gross_amount / gas_m3

    @property
    def extra_state_attributes(self):
        attrs = dict()
        if self._state is not None:
            attrs["last_invoice_date"] = self._state.paying_deadline_date
            attrs["last_invoice_gross_amount"] = self._state.gross_amount
            attrs["last_invoice_wear_m3"] = self._state.wear_m3
            attrs["last_invoice_wear_KWH"] = self._state.wear_kwh
            attrs["last_invoice_number"] = self._state.number
        return attrs


class PgnigReadingDateSensor(PgnigBaseSensor):
    """When the meter reading was taken, as opposed to when it was fetched.

    EBOK publishes a reading days after the meter records it. Attributes on the
    meter sensor carry the date, but an attribute cannot be graphed, alerted on,
    or shown as "5 days ago" in the UI. As a timestamp entity it can.
    """

    _name_prefix = "Orlen Gas Reading Date"
    _unique_id_prefix = "pgnig_reading_date"

    _attr_device_class = SensorDeviceClass.TIMESTAMP

    def _state_from(self, data: PgnigData):
        return data.readings.get(self.meter_id)

    @property
    def state(self):
        if self._state is None:
            return None
        return _as_utc(self._state.reading_date_utc)

    @property
    def extra_state_attributes(self):
        if self._state is None:
            return {}
        return {
            "reading_type": self._state.type,
            "reading_status": self._state.status,
            "value": self._state.value,
        }


class PgnigMarginalPriceSensor(PgnigBaseSensor):
    """Gas price per m³ with the standing charges taken out.

    PgnigCostTrackingSensor reports gross_amount / volume, which includes the
    subscription and fixed distribution fee and therefore rises as consumption
    falls. This entity reports the fitted rate instead, which is what the Energy
    dashboard needs as a price.
    """

    _name_prefix = "Orlen Gas Marginal Price"
    _unique_id_prefix = "pgnig_marginal_price"

    _attr_native_unit_of_measurement = "PLN/m³"
    _attr_device_class = SensorDeviceClass.MONETARY
    _attr_state_class = SensorStateClass.MEASUREMENT

    def _state_from(self, data: PgnigData):
        invoices = priced_invoices(data.invoices, self.id_local)
        if not invoices:
            return None
        return {
            "fit": _fit_marginal_price(invoices),
            "latest": max(invoices, key=lambda z: z.date),
        }

    @property
    def state(self):
        fit = (self._state or {}).get("fit")
        if fit is None:
            return None
        return round(fit[0], 4)

    @property
    def extra_state_attributes(self):
        fit = (self._state or {}).get("fit")
        if fit is None:
            return {}
        _, standing, used, model, residual = fit
        attrs = {
            "invoices_used": used,
            "model": model,
        }
        if residual is not None:
            attrs["residual_pct"] = round(residual, 2)
        if model == "per_day":
            attrs["fixed_charge_per_day"] = round(standing, 4)
            latest = self._state.get("latest")
            days = _billing_days(latest) if latest is not None else None
            if days:
                attrs["fixed_charge_per_invoice"] = round(standing * days, 2)
        else:
            attrs["fixed_charge_per_invoice"] = round(standing, 2)
        return attrs
