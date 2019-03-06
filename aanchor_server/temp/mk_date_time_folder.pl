#!/usr/local/bin/perl
use POSIX qw(strftime);
use Cwd;
my $dir = getcwd;

my $datestring = strftime "%a_%b_%e_%H_%M_%S_%Y", localtime;
printf("date and time - $datestring\n");
my $newfolder = "$dir/$datestring";
printf($newfolder);
mkdir $newfolder
