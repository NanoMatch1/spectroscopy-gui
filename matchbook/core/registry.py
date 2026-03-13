"""Module registry — discovers, validates, and registers analysis modules.

The registry is the glue layer: it takes a module (any object satisfying the
``AnalysisModule`` protocol), pulls out its descriptors, wires up the
pipeline, and optionally hands the module a context for GUI integration.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

from matchbook.core.data_service import DataService
from matchbook.core.module_base import (
    AnalysisModule,
    DataGroupDescriptor,
    PipelineStepDescriptor,
)
from matchbook.core.pipeline import Pipeline, PipelineStep

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module context (handed to modules at registration)
# ---------------------------------------------------------------------------

@dataclass
class ModuleContext:
    """Thin façade provided to modules at registration time.

    GUI callback fields are ``None`` when running headless.
    """
    data_service: DataService
    pipeline: Pipeline
    request_sidebar_section: Callable[[DataGroupDescriptor], None] | None = None
    request_parameter_widget: Callable[[PipelineStepDescriptor], None] | None = None


# ---------------------------------------------------------------------------
# Registered module record
# ---------------------------------------------------------------------------

@dataclass
class RegisteredModule:
    """Everything the framework remembers about a registered module."""
    module: Any  # the original module object
    name: str
    display_name: str
    data_groups: list[DataGroupDescriptor]
    pipeline_step_descriptors: list[PipelineStepDescriptor]
    pipeline: Pipeline


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class Registry:
    """Discovers, validates, and registers analysis modules."""

    def __init__(self, data_service: DataService) -> None:
        self._data_service = data_service
        self._modules: dict[str, RegisteredModule] = {}

        # GUI callbacks — set by the GUI layer after construction.
        self.on_sidebar_section: Callable[[DataGroupDescriptor], None] | None = None
        self.on_parameter_widget: Callable[[PipelineStepDescriptor], None] | None = None

    # -- registration ----------------------------------------------------------

    def register(self, module: Any) -> RegisteredModule:
        """Validate and register an analysis module.

        Parameters
        ----------
        module:
            Any object satisfying the ``AnalysisModule`` protocol — it must
            expose ``name``, ``display_name``, ``data_groups()``,
            ``pipeline_step_descriptors()``, and ``create_pipeline_steps()``.

        Returns
        -------
        RegisteredModule
            The registration record, also stored internally.

        Raises
        ------
        TypeError
            If *module* does not satisfy the protocol.
        """
        if not isinstance(module, AnalysisModule):
            raise TypeError(
                f"Module {module!r} does not satisfy the AnalysisModule protocol.  "
                f"Required attributes/methods: name, display_name, data_groups(), "
                f"pipeline_step_descriptors(), create_pipeline_steps()."
            )

        mod_name: str = module.name
        mod_display: str = module.display_name
        groups: list[DataGroupDescriptor] = module.data_groups()
        step_descs: list[PipelineStepDescriptor] = module.pipeline_step_descriptors()

        # Build pipeline for this module
        pipeline = Pipeline()
        raw_steps = module.create_pipeline_steps()
        for step in raw_steps:
            if not isinstance(step, PipelineStep):
                raise TypeError(
                    f"create_pipeline_steps() must return PipelineStep instances, "
                    f"got {type(step).__name__}"
                )
            pipeline.add_step(step)

        record = RegisteredModule(
            module=module,
            name=mod_name,
            display_name=mod_display,
            data_groups=groups,
            pipeline_step_descriptors=step_descs,
            pipeline=pipeline,
        )
        self._modules[mod_name] = record

        logger.info(
            f"Registered module '{mod_display}' ({mod_name}): "
            f"{len(groups)} data groups, {len(step_descs)} pipeline steps"
        )

        # Notify GUI if callbacks are wired
        if self.on_sidebar_section:
            for group in groups:
                self.on_sidebar_section(group)
        if self.on_parameter_widget:
            for desc in step_descs:
                self.on_parameter_widget(desc)

        # Provide context to the module if it has an on_register hook
        if hasattr(module, "on_register") and callable(module.on_register):
            ctx = ModuleContext(
                data_service=self._data_service,
                pipeline=pipeline,
                request_sidebar_section=self.on_sidebar_section,
                request_parameter_widget=self.on_parameter_widget,
            )
            module.on_register(ctx)

        return record

    # -- lookup ----------------------------------------------------------------

    def get_module(self, name: str) -> RegisteredModule:
        """Return the registered module record for *name*."""
        return self._modules[name]

    def get_pipeline(self, module_name: str) -> Pipeline:
        """Return the pipeline for *module_name*."""
        return self._modules[module_name].pipeline

    @property
    def registered_modules(self) -> dict[str, RegisteredModule]:
        """All registered modules, keyed by module name."""
        return dict(self._modules)

    @property
    def all_data_groups(self) -> list[DataGroupDescriptor]:
        """Merged data groups from all modules, sorted by order."""
        groups: list[DataGroupDescriptor] = []
        for record in self._modules.values():
            groups.extend(record.data_groups)
        groups.sort(key=lambda g: g.order)
        return groups
