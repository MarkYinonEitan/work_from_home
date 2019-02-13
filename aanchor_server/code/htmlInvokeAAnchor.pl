#!/usr/bin/perl -w
my $file = "/specific/a/home/cc/cs/ppdock/webserver/AAnchor/temp/temp0.txt";


unless(open FILE, '>'.$file) {
    # Die with error message
    # if we can't open it.
    die "\nUnable to create $file\n";
}

# Write some text to the file.
print FILE "000\n";

use strict;
use CGI qw/:standard/;
use Fcntl qw (:flock);

print FILE "001\n";

my $home = "/specific/a/home/cc/cs/ppdock/webserver/AAnchor/bin";
#MARK require "/specific/a/home/cc/cs/ppdock/webserver/AAnchor/temp/UTIL.pm";
##new##
#MARK require "$home/workspace/project/TemplateDocking/TemplateDocking_code/PDB.pm";

my $upload_dir = "/specific/disk1/webservertmp/AAnchor/upload";
my $runs_dir = "/specific/disk1/webservertmp/AAnchor/runs/";



my $serverlog = "/specific/disk1/webservertmp/AAnchor/server.log";
my $debug = "/specific/disk1/webservertmp/AAnchor/debug.log";
my $aanchor_dir = "/specific/a/home/cc/cs/ppdock/webserver/AAnchor/";
my $code_dir = $aanchor_dir."/bin/";

my $chimera_script = $aanchor_dir."/bin/pythoncode/server/create_input_file_chimera.py";

my $python_script = $aanchor_dir."/bin/pythoncode/server/run_server.py";

my $query = new CGI;
print $query->header();

print FILE "002\n";

my $em_file = $query->param("em_file");
$em_file =~ s/^\s+//;
print FILE "$em_file\n";
print FILE "003\n";


my $resolution = $query->param("resolution");

print FILE "resolution\n";
print FILE "$resolution\n";
print FILE "004\n";

my $t1 =  lc($query->param("em_file"));
print FILE "$t1\n";
print FILE "005\n";


if (length $em_file > 0) {

	$em_file =~ s/.*[\/\\](.*)/$1/;

	my $rupload_filehandle = $query->upload("em_file");
	print FILE "7\n";


	print FILE "$rupload_filehandle \n";
	print FILE "8\n";
	open UPLOADFILE, ">$upload_dir/$em_file";
	while ( <$rupload_filehandle> ) {
		print UPLOADFILE;
		}
	close UPLOADFILE;
}


print FILE "UPLOAD FINISHED\n";
print FILE "006\n";

chdir $aanchor_dir;


my $input_map_file = "$upload_dir/$em_file";
print FILE "INP FILE\n";
print FILE "$input_map_file\n";

my $output_gz_file = "$upload_dir/$em_file.gz.pkl";
print FILE "OUT FILE\n";
print FILE "$output_gz_file\n";

my $cmd = "chimera-1.13 --nogui $chimera_script $input_map_file $output_gz_file";
`$cmd >> $debug 2>&1 `;
print FILE "MAP READED AND CONERTED\n";




my $cmd1 = "source $aanchor_dir/virt_env/bin/activate";

my $cmd2 = "export KERAS_BACKEND=theano";

my $cmd3 = "python $python_script $output_gz_file $resolution 'mail_id' ";

my $cmd_all = "$cmd1 >> $debug 2>&1 && $cmd2 >> $debug 2>&1 && $cmd3 >> $debug 2>&1";

`$cmd_all  `;
print FILE "PYTHON RUN\n";

`deactivate 2>&1 `;




print FILE "end\n";
close FILE
