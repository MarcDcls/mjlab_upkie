% scale(1000) import("mj5208_support.stl");

// Append pure shapes (cube, cylinder and sphere), e.g:
// cube([10, 10, 10], center=true);
// cylinder(r=10, h=10, center=true);
// sphere(10);

translate([0, 60.4, 6.25])
cylinder(r=22, h=60.5, center=true);

translate([19, 26.5, 6.25])
cylinder(r=21, h=60.5, center=true);

translate([-19, 26.5, 6.25])
cylinder(r=21, h=60.5, center=true);