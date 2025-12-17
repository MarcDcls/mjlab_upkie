% scale(1000) import("qdd100_stator.stl");

// Append pure shapes (cube, cylinder and sphere), e.g:
// cube([10, 10, 10], center=true);
// cylinder(r=10, h=10, center=true);
// sphere(10);

translate([-14.35, 0, -65])
rotate([0, 90, 0])
cylinder(r=50, h=28.7, center=true);