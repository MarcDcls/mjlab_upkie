% scale(1000) import("qdd100_support.stl");

// Append pure shapes (cube, cylinder and sphere), e.g:
// cube([10, 10, 10], center=true);
// cylinder(r=10, h=10, center=true);
// sphere(10);


translate([0, 0, -12.5])
cylinder(r=50, h=25, center=true);

translate([0, 0, -43])
rotate([90, 90, 0])
cylinder(r=20, h=60, center=true);
