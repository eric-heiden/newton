# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np

from .unittest_utils import assert_np_equal


def compare_mujoco_specs(spec1, spec2, tol=1e-6):
    assert len(spec1.bodies) == len(spec2.bodies), (
        f"Number of bodies mismatch {len(spec1.bodies)} != {len(spec2.bodies)}"
    )
    assert len(spec1.joints) == len(spec2.joints), (
        f"Number of joints mismatch {len(spec1.joints)} != {len(spec2.joints)}"
    )

    # assert len(spec1.geoms) == len(spec2.geoms), "Number of geoms mismatch"
    # assert len(spec1.actuators) == len(spec2.actuators), "Number of actuators mismatch"

    for i in range(len(spec1.bodies)):
        body1 = spec1.bodies[i]
        body2 = spec2.bodies[i]
        assert body1.name == body2.name, f"Body {i} name mismatch {body1.name} != {body2.name}"
        # assert_np_equal(body1.pos, body2.pos, tol=tol)
        # assert_np_equal(body1.quat, body2.quat, tol=tol)
        assert abs(body1.mass - body2.mass) < tol, f"Body {i} mass mismatch {body1.mass} != {body2.mass}"
        if not np.isnan(body1.fullinertia[0]) and not np.isnan(body2.fullinertia[0]):
            # compare full inertia directly
            assert_np_equal(body1.fullinertia, body2.fullinertia, tol=tol)
        else:
            # compare inertia components as they are available
            inertia1 = body1.fullinertia[:3] if not np.isnan(body1.fullinertia[0]) else body1.inertia
            inertia2 = body2.fullinertia[:3] if not np.isnan(body2.fullinertia[0]) else body2.inertia
            assert_np_equal(inertia1, inertia2, tol=tol)
        assert_np_equal(body1.ipos, body2.ipos, tol=tol)
        assert_np_equal(body1.iquat, body2.iquat, tol=tol)

    for i in range(len(spec1.joints)):
        assert spec1.joints[i].name == spec2.joints[i].name, (
            f"Joint {i} name mismatch {spec1.joints[i].name} != {spec2.joints[i].name}"
        )
        assert spec1.joints[i].type == spec2.joints[i].type, (
            f"Joint {i} type mismatch {spec1.joints[i].type} != {spec2.joints[i].type}"
        )
        assert spec1.joints[i].group == spec2.joints[i].group, (
            f"Joint {i} group mismatch {spec1.joints[i].group} != {spec2.joints[i].group}"
        )
        assert spec1.joints[i].dof == spec2.joints[i].dof, (
            f"Joint {i} dof mismatch {spec1.joints[i].dof} != {spec2.joints[i].dof}"
        )
        assert spec1.joints[i].stiffness == spec2.joints[i].stiffness, (
            f"Joint {i} stiffness mismatch {spec1.joints[i].stiffness} != {spec2.joints[i].stiffness}"
        )
        assert spec1.joints[i].damping == spec2.joints[i].damping, (
            f"Joint {i} damping mismatch {spec1.joints[i].damping} != {spec2.joints[i].damping}"
        )
        assert spec1.joints[i].frictionloss == spec2.joints[i].frictionloss, (
            f"Joint {i} frictionloss mismatch {spec1.joints[i].frictionloss} != {spec2.joints[i].frictionloss}"
        )

    return True
