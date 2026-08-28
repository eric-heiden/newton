Fix XPBD cloth contact friction so it obeys Coulomb's law. The tangential impulse is now accumulated over the step and applied incrementally, instead of being re-applied at its full Coulomb bound on every solver iteration, so friction is a material property rather than a function of `iterations`. The positional and velocity passes also share one per-step friction budget instead of each spending a full one, which previously doubled the achievable friction force. A cloth patch on an incline now holds exactly when `mu` exceeds `tan(theta)`.

Fix XPBD cloth triangle-intersection recovery applying one unaveraged correction per intersecting pair, so a vertex shared by several pairs was displaced far past separation and created more intersections than it removed. Recovery now uses a shared Jacobi scale.

Fix XPBD cloth vertices repelling their own immediate mesh neighbours through the particle-particle contact fallback whenever vertex spacing was below the particle diameter, which tore resting cloth apart. Mesh vertices that overlap in the rest pose no longer generate contacts; free particles still collide when they start overlapped.

Fix the cloth bending example being tested while the sheet was still falling.
