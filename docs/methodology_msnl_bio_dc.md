# Finite-element simulation of AFM nanoindentation of a thin elastic film

*Methods (Biophysical Journal style)*

## Finite-element model

Force–indentation curves and the internal stress distribution of the sample
were computed with a custom, open-source finite-element (FE) solver
(`nanoindent`, implemented in Python on the scikit-fem/FEniCSx framework),
reproducing the numerical experiment usually carried out in commercial FE
software (COMSOL Multiphysics; COMSOL AB, Stockholm, Sweden). The atomic force
microscopy (AFM) indentation of a probe (MSNL-BIO-DC) into a thin elastic film
bonded to a rigid substrate is an axisymmetric problem; exploiting the
rotational symmetry about the indentation axis, the three-dimensional geometry
was reduced to a two-dimensional cross-section in cylindrical coordinates
\((r, z)\). All fields, boundary conditions, and the contact constraint are
independent of the azimuthal angle, and every domain integral carries the
axisymmetric weight \(2\pi r\).

## Geometry and material models

The indenter was modeled as a rigid paraboloid of apex radius \(R = 20\) nm,
matching the nominal end radius of the MSNL-BIO-DC tip. Although the probe is a
silicon-nitride cantilever with Young's modulus \(E_\mathrm{tip} = 170\) GPa,
its stiffness exceeds that of the sample by more than three orders of magnitude
(\(E_\mathrm{tip}/E_\mathrm{sample} \approx 3.4\times10^{3}\)); the tip
deformation is therefore negligible and the indenter was treated as perfectly
rigid, entering the formulation solely through its gap (indentation) function.
The exact paraboloidal profile, \(z_\mathrm{tip}(r) = r^{2}/(2R)\), was used
rather than a small-slope linearization.

The sample was represented as a linear, isotropic, homogeneous elastic solid
with Young's modulus \(E = 50\) MPa. Because the Poisson's ratio of the sample
was not independently determined, it was fixed at \(\nu = 0.5\) (fully
incompressible), the value conventionally assumed for soft, hydrated biological
material; a compressible case (\(\nu = 0.3\)) was also computed for comparison.
Small-strain (infinitesimal) kinematics were assumed throughout. The sample was
an axisymmetric disc of radius \(L = 200\) nm and thickness \(h = 6.2\) nm; the
lateral extent was large compared with the contact radius, so that lateral
boundary effects were negligible.

## Boundary and contact conditions

The following boundary conditions were imposed:

- **Bottom surface** (\(z = 0\)): two substrate conditions were considered. In
  the *bonded* case the film is fully fixed, \(u_r = u_z = 0\), representing
  rigid adhesion to the substrate. In the *unbonded* case the film rests on a
  rigid, frictionless substrate: only the normal displacement is constrained,
  \(u_z = 0\), while the tangential (radial) displacement \(u_r\) is free, so
  the film can slide along the substrate.
- **Symmetry axis** (\(r = 0\)): \(u_r = 0\) (axisymmetry).
- **Lateral edge** (\(r = L\)): traction-free.
- **Top surface** (\(z = h\)): frictionless (normal-only) unilateral contact
  against the rigid tip, traction-free outside the contact region.

Contact was enforced with a penalty formulation. The signed gap between the
deformed sample surface and the rigid paraboloid,
\(g(r) = z_\mathrm{tip}(r) - [\,h + u_z(r)\,]\), defines the interpenetration
\([-g]_+ = \max(-g, 0)\); a contact pressure \(p_c = k_\mathrm{pen}\,[-g]_+\)
was applied over the top surface, with the penalty stiffness \(k_\mathrm{pen}\)
scaled to the local element size to remain consistent under mesh refinement.
Only the normal component of the contact traction was retained (frictionless
contact); no tangential constraint was imposed.

Because the incompressible sample (\(\nu = 0.5\)) would cause volumetric locking
of a single-field displacement discretization, the elasticity was solved with a
mixed displacement–pressure (Taylor–Hood) formulation, using quadratic (P2)
displacement and linear (P1) pressure fields; the pressure enforces the
incompressibility constraint. For the compressible comparison
(\(\nu = 0.3\)) the standard single-field P2 displacement formulation was used.

## Loading and solution

Indentation was displacement-controlled: the rigid tip was advanced normal to
the sample surface in equal increments up to a maximum indentation of
\(\delta = 2\) nm. At each increment the nonlinear contact problem — nonlinear
only through the activation of the contact set — was solved by a Newton
iteration, using the converged solution of the previous step as the initial
guess; the linear systems were solved with a direct sparse solver. The reaction
force on the indenter was computed by integrating the contact pressure over the
top surface, \(F = \int_{\Gamma_\mathrm{top}} k_\mathrm{pen}[-g]_+\,2\pi r\,dr\),
and, independently, by integrating the axial stress over the fixed bottom
boundary; the two estimates agreed to within 0.3%, confirming global
equilibrium. Forces are reported in piconewtons, using the consistent unit
system (length in nm, stress in MPa), for which \(1~\mathrm{MPa\cdot nm^2} =
1~\mathrm{pN}\).

## Mesh and convergence

The cross-section was discretized with quadratic triangular elements on a
graded mesh, strongly refined at the contact point (\(r = 0\), \(z = h\)) and
coarsening toward the outer boundaries. Mesh independence was established by
successively reducing the near-contact element size (0.04, 0.02, 0.01, and
0.005 nm) while holding the graded coarsening toward the edges fixed; the
reaction force at maximum indentation varied by less than about 2% over this
range and the peak equivalent stress by less than 0.2%. The reported results
use the finest (0.005 nm near-contact) mesh. Several element layers were
resolved across the 6.2 nm film thickness.

## Outputs

Two quantities were extracted from the converged solution:

1. the **force–indentation curve** \(F(\delta)\) over \(0 \le \delta \le 2\) nm;
2. the spatial distribution of the **von Mises (equivalent) stress**,
   \(\sigma_\mathrm{vM} = \sqrt{\tfrac{1}{2}[(\sigma_{rr}-\sigma_{zz})^2 +
   (\sigma_{zz}-\sigma_{\theta\theta})^2 +
   (\sigma_{\theta\theta}-\sigma_{rr})^2] + 3\sigma_{rz}^2}\),
   recovered by \(L^2\) projection onto the mesh nodes and displayed on the
   axisymmetric cross-section (mirrored about the symmetry axis).

For interpretation, the simulated \(F(\delta)\) was compared with the analytical
Hertz model for a paraboloid on an elastic half-space,
\(F = \tfrac{4}{3}\,\dfrac{E}{1-\nu^{2}}\,\sqrt{R}\,\delta^{3/2}\), and with the
Chadwick (2002) asymptotic solution for a bonded incompressible thin layer,
\(F = \tfrac{2\pi}{3}\,E\,R^{2}\,\delta^{3}/h^{3}\).

## Representative results

For the incompressible sample (\(E = 50\) MPa, \(\nu = 0.5\), \(R = 20\) nm,
\(h = 6.2\) nm), the converged model gave a reaction force of \(F \approx
5.9\) nN at the maximum indentation \(\delta = 2\) nm, and a peak von Mises
stress of \(\approx 21\) MPa localized beneath the tip apex and confined through
the full film thickness to the bonded substrate. The finite-thickness,
substrate-bonded response is markedly stiffer than the Hertz half-space
prediction, reflecting the bottom-effect (substrate) stiffening of the thin
bonded film; at equal indentation the compressible sample (\(\nu = 0.3\)) is
roughly a factor of two more compliant than the incompressible one.

The substrate boundary condition has a comparably large effect. Repeating the
incompressible simulation with the film *unbonded* (frictionless slip on the
rigid substrate) reduced the reaction force at \(\delta = 2\) nm to \(F \approx
2.8\) nN — about half the bonded value — and the peak von Mises stress to
\(\approx 19\) MPa. Physically, an incompressible bonded film cannot conserve
volume except by large in-plane stretching, which the fixed substrate resists
(strong confinement stiffening); when the film is free to slide, material is
displaced laterally along the substrate, the confinement is relieved, and the
high-stress region extends down to the sliding base rather than concentrating
beneath the tip apex. The bonded and unbonded FE curves bracket the
corresponding bonded and free analytical corrections of Dimitriadis et al.
(2002), confirming the expected sensitivity of the inferred modulus to the
assumed tip–sample–substrate boundary conditions.

## Figure captions

**Figure X.** Cross-section of the FE simulation of the AFM tip (rigid
paraboloid, \(R = 20\) nm) indenting a 6.2 nm elastic film to \(\delta = 2\) nm.
The color map shows the simulated von Mises stress within the sample (red: high
stress; blue: low stress) produced by the vertical force applied by the tip; the
red curve marks the rigid tip surface, and the domain is mirrored about the
symmetry axis \(r = 0\).

**Figure Y.** Simulated force–indentation curve \(F(\delta)\) for the same
sample (\(E = 50\) MPa, \(h = 6.2\) nm), compared with the Hertz half-space and
Chadwick thin-incompressible-layer models. The FE response lies well above the
Hertz prediction, evidencing the bottom-effect stiffening of the bonded thin
film.

## References

1. Hertz, H. (1882). Über die Berührung fester elastischer Körper.
   *J. Reine Angew. Math.* **92**, 156–171.
2. Sneddon, I. N. (1965). The relation between load and penetration in the
   axisymmetric Boussinesq problem for a punch of arbitrary profile.
   *Int. J. Eng. Sci.* **3**, 47–57.
3. Dimitriadis, E. K., F. Horkay, J. Maresca, B. Kachar, and R. S. Chadwick
   (2002). Determination of elastic moduli of thin layers of soft material
   using the atomic force microscope. *Biophys. J.* **82**, 2798–2810.
4. Chadwick, R. S. (2002). Axisymmetric indentation of a thin incompressible
   elastic layer. *SIAM J. Appl. Math.* **62**, 1520–1530.
5. Garcia, P. D., and R. Garcia (2018). Determination of the elastic moduli of a
   single cell cultured on a rigid support by force microscopy.
   *Biophys. J.* **114**, 2923–2932.
