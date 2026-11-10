use strict;
local $/;
my %crlf;
sub load { my ($p) = @_; open(my $in, '<:raw', $p) or die $p; my $s = <$in>; close $in; $crlf{$p} = ($s =~ /\r\n/) ? 1 : 0; $s =~ s/\r\n/\n/g; return $s; }
sub save { my ($p, $s) = @_; $s =~ s/\n/\r\n/g if $crlf{$p}; open(my $out, '>:raw', $p) or die; print $out $s; close $out; }
sub sub1 { my ($sref, $from, $to) = @_; my $i = index($$sref, $from); die "missing: $from" if $i < 0; substr($$sref, $i, length($from)) = $to; }

my $a = load('Sources/SAGAVulkan/VulkanRenderABI.swift');
sub1(\$a, q~        static let vertex: UInt32 = 0x1~, q~        static let vertex: UInt32 = 0x1
        static let task: UInt32 = 0x40
        static let mesh: UInt32 = 0x80~);
save('Sources/SAGAVulkan/VulkanRenderABI.swift', $a);

my $r = load('Sources/SAGAVulkan/VulkanRender.swift');
sub1(\$r, q~        var viewMask: UInt32?
        var layer: UInt32 = 0~, q~        var viewMask: UInt32?
        /// Mesh task group counts; the plan then ends in vkCmdDrawMeshTasksEXT instead of a vertex draw.
        var meshGroups: (UInt32, UInt32, UInt32)?
        var layer: UInt32 = 0~) if 0;
save('Sources/SAGAVulkan/VulkanRender.swift', $r);
print "abi ok\n";
