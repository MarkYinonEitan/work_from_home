#!/usr/bin/env python

"""Send the contents of a directory as a MIME message."""

import os
import sys
import smtplib
# For guessing MIME type based on file name extension
import mimetypes
import zipfile
from optparse import OptionParser
import tempfile
import shutil

from email import encoders
from email.message import Message
from email.mime.audio import MIMEAudio
from email.mime.base import MIMEBase
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.Utils import COMMASPACE, formatdate
from email import Encoders

COMMASPACE = ', '


def main():
	parser = OptionParser(usage="""\
Send the contents of a directory as a MIME message.

Usage: %prog [options]

Unless the email is sent by forwarding to your local
SMTP server, which then does the normal delivery process.  Your local machine
must be running an SMTP server.
""")
	parser.add_option('-d', '--directory',
                      type='string', action='store',
                      help="""Mail the contents of the specified directory,
                      otherwise use the current directory.  Only the regular
                      files in the directory are sent, and we don't recurse to
                      subdirectories.""")

	parser.add_option('-r', '--recipient',
                      type='string', action='append', metavar='RECIPIENT',
                      default=[], dest='recipients',
                      help='A To: header value (at least one required)')
	opts, args = parser.parse_args()

	if not opts.recipients or not opts.directory:
		parser.print_help()
		sys.exit(1)

	zip_and_send_mail(opts.recipients,opts.directory,"","")


def zip_and_send_mail(email_ad, work_dir):

	out_name = 'AANCHOR' + os.path.basename(os.path.normpath(work_dir))

	print "send mail to "+ str(email_ad) + " folder "+work_dir

	ziped_file = os.path.join(work_dir, out_name)

	shutil.make_archive(ziped_file, "gztar",work_dir)

	print "send file "+ziped_file+".tar.gz"

	send_mail(ziped_file+".tar.gz", email_ad)


def send_mail(ziped_file, recipients):
	msg = MIMEMultipart()

	msg['From'] =  'ppdock@tau.ac.il'


	msg['To'] = COMMASPACE.join(recipients)
 	msg['Date'] = formatdate(localtime=True)
  	msg['Subject'] = "AAnchor results: "
	msg.attach( MIMEText( '\n Thank you for using AAnchor! \n ***********************************\n \nYour AAnchor run has just finished. Result files are attached.\n\n For any questions / comments / problems please contact: markroza@tau.ac.il \n ') )
	part = MIMEBase('application', "octet-stream")
	part.set_payload(open(ziped_file, "rb").read())
    	Encoders.encode_base64(part)

    	part.add_header('Content-Disposition', 'attachment; filename="%s"' % os.path.basename(ziped_file))
    	msg.attach(part)

  	smtp = smtplib.SMTP('smtp.gmail.com',587)
	smtp.starttls()
	smtp.login('marik.s79@gmail.com','Marik29gmail')
  	smtp.sendmail('marik.s79@gmail.com', recipients, msg.as_string())
  	smtp.close()
	print 'done!'

def zipdir(path, ziph, out_folder):
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(path, file), os.path.join(out_folder, file))

if __name__ == '__main__':
    main()
