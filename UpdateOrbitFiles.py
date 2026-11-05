""" Updates asteroid and comet orbit files from MPC and JPL website. """

from __future__ import print_function, division, absolute_import

import os
import re
import sys
import ssl
import socket
import shutil
import datetime
import unicodedata

# Fix certificates error
ssl._create_default_https_context = ssl._create_unverified_context


if sys.version_info.major < 3:
	import urllib as urllibrary
	from HTMLParser import HTMLParser

else:
	import urllib.request as urllibrary
	from html.parser import HTMLParser

def updateOrbitFiles():
	""" Updates asteroid and comet orbit files from MPC and JPL website. """

	# Set a 15 second connection timeout so the compile is not held by a hanging download
	socket.setdefaulttimeout(15)

	# Comets
	comets_url = "https://ssd.jpl.nasa.gov/dat/ELEMENTS.COMET"

	# Amors
	amors_url = "http://cgi.minorplanetcenter.net/cgi-bin/textversion.cgi?f=lists/Amors.html"

	# Apollos
	apollos_url = "http://cgi.minorplanetcenter.net/cgi-bin/textversion.cgi?f=lists/Apollos.html"

	# Atens
	atens_url = "http://cgi.minorplanetcenter.net/cgi-bin/textversion.cgi?f=lists/Atens.html"


	dir_path = os.path.split(os.path.abspath(__file__))[0]
	dir_path = os.path.join(dir_path, 'wmpl', 'share')

	file_names = ['ELEMENTS.COMET', 'Amors.txt', 'Apollos.txt', 'Atens.txt']
	url_list = [comets_url, amors_url, apollos_url, atens_url]

	# Temporary file path
	temp_path = os.path.join(dir_path, "temp.part")

	# Download all orbit files
	for fname, url in zip(file_names, url_list):
		print('Downloading {:s}...'.format(fname))

		tries = 0
		while tries < 5:

			# Download the file to a temporary location (as not to overwrite the existing file)
			try:

				# Download the file to a temporary location
				urllibrary.urlretrieve(url, temp_path)

				# Move the downloaded file to a final location
				shutil.move(temp_path, os.path.join(dir_path, fname))

				break

			except Exception as e:
				print("Download failed with:" + repr(e))
				tries += 1


		print(' ... done!')




# Columns of wmpl/share/Meteorites_with_Orbits.csv
METEORITES_HEADER = u"# Ab.;Name;Type;Country;Date (UTC);Time (UTC);Time Zone;Mass (kg);a (AU);a error (AU);e;" \
	u"e error (AU);i (\u00b0);i error (\u00b0);Arg. Peri. (\u00b0);Arg. Peri. Error (\u00b0);Asc. node (\u00b0);" \
	u"Asc. node error (\u00b0);Ref."



class MeteoriteTableParser(HTMLParser):
	""" Collect the cells of all tables in an HTML page. """

	def __init__(self):
		HTMLParser.__init__(self)

		# List of tables, each a list of rows of cell strings
		self.tables = []

		# Table, row and cell which are being read
		self.rows = None
		self.row = None
		self.cell = None



	def handle_starttag(self, tag, attrs):

		if tag == 'table':
			self.rows = []

		elif (tag == 'tr') and (self.rows is not None):
			self.row = []

		elif (tag in ('td', 'th')) and (self.row is not None):
			self.cell = ""

		# Line breaks inside a cell separate the value and its error
		elif (tag == 'br') and (self.cell is not None):
			self.cell += " "



	def handle_endtag(self, tag):

		if (tag in ('td', 'th')) and (self.cell is not None):
			self.row.append(re.sub(r'\s+', ' ', self.cell).strip())
			self.cell = None

		elif (tag == 'tr') and (self.row is not None):
			self.rows.append(self.row)
			self.row = None

		elif (tag == 'table') and (self.rows is not None):
			self.tables.append(self.rows)
			self.rows = None



	def handle_data(self, data):

		if self.cell is not None:
			self.cell += data



def meteoriteTableRows(html_text):
	""" Extract the rows of the meteorite table from the HTML of meteoriteorbits.info.

	Arguments:
		html_text: [str] HTML of the page.

	Return:
		[list] Rows of the table, each a list of cell strings. The first row is the header. Empty if the table
			was not found.
	"""

	parser = MeteoriteTableParser()
	parser.feed(html_text)

	# The meteorite table is the one with the orbital elements in its header
	for rows in parser.tables:
		if rows and any('a (AU)' in cell for cell in rows[0]):
			return rows

	return []



def tableColumn(header, key):
	""" Find the column of the table whose header starts with the given key.

	Arguments:
		header: [list] Lower case cells of the header row.
		key: [str] Lower case start of the column name.

	Return:
		[int] Index of the column, None if no column was found.
	"""

	for i, name in enumerate(header):
		if name.startswith(key):
			return i

	return None



def valueError(cell):
	""" Split a table cell such as "1.844 +/-0.003" into its value and error.

	Arguments:
		cell: [str] Cell of the table.

	Return:
		(value, err): [tuple of str] The value and its error. ("???", "") for an unknown value, and
			(None, None) if the cell cannot be read.
	"""

	# Replace the typographic minus and remove the dashes and footnote marks
	cell = cell.replace(u'\u2212', '-').replace(u'\u2013', '').replace('*', '').strip()

	if (not cell) or (cell == '???'):
		return "???", ""

	# Read a symmetric error, e.g. "1.844 +/-0.003"
	match = re.match(u'^([-+]?\\d+(?:\\.\\d+)?)\\s*(?:\u00b1\\s*(\\d+(?:\\.\\d+)?))?$', cell)
	if match:

		# An error of zero means that none was given
		err = match.group(2) or ""
		if err and (float(err) == 0):
			err = ""

		return match.group(1), err

	# Read an asymmetric error, e.g. "2.62 +0.7-0.46"
	match = re.match(r'^([-+]?\d+(?:\.\d+)?)\s*\+(\d+(?:\.\d+)?)\s*-(\d+(?:\.\d+)?)$', cell)
	if match:
		return match.group(1), "+{:s}/-{:s}".format(match.group(2), match.group(3))

	return None, None



def meteoriteIdentity(line):
	""" Identify a meteorite in a line of the meteorite table. Two lines are the same meteorite if either the
		names or the abbreviations with the dates match, so corrected spellings are not taken as new
		meteorites.

	Arguments:
		line: [str] Line of the table.

	Return:
		(name, ident):
			- name: [str] Name without accents, case and punctuation.
			- ident: [tuple] Abbreviation and date.
	"""

	fields = line.split(';')

	# Remove the accents, case and punctuation from the name
	name = unicodedata.normalize('NFKD', fields[1])
	name = "".join(c.lower() for c in name if c.isalnum() and not unicodedata.combining(c))

	return name, (fields[0], fields[4])



def changedMeteorites(file_lines, new_lines):
	""" Find the meteorites of the existing table whose downloaded line is different.

	Arguments:
		file_lines: [list] Lines of the existing table.
		new_lines: [list] Lines of the downloaded table.

	Return:
		[list] Tuples (index in file_lines, existing line, downloaded line).
	"""

	# Index the existing meteorites by name and by abbreviation and date
	by_name = {}
	by_ident = {}
	for i, line in enumerate(file_lines):
		name, ident = meteoriteIdentity(line)
		by_name[name] = i
		by_ident[ident] = i

	changed = []
	for line in new_lines:

		name, ident = meteoriteIdentity(line)
		i = by_name.get(name, by_ident.get(ident))

		if (i is not None) and (file_lines[i] != line):
			changed.append((i, file_lines[i], line))

	return changed



def askUpdates(changed):
	""" Show how the downloaded values differ from the existing ones and ask which meteorites to update. The
		default answer is to keep the existing line.

	Arguments:
		changed: [list] Tuples (index, existing line, downloaded line) from changedMeteorites().

	Return:
		[list] The tuples of the meteorites to update.
	"""

	field_names = [name.lstrip('# ') for name in METEORITES_HEADER.split(';')]

	print()
	print("{:d} meteorites in the file differ from the list. The file is kept unless you say yes.".format(\
		len(changed)))

	updates = []
	for k, (i, old, new) in enumerate(changed):

		old_fields = old.split(';')
		new_fields = new.split(';')

		# Show the fields which differ
		print()
		print("[{:d}/{:d}] {:s} ({:s})".format(k + 1, len(changed), old_fields[1], old_fields[0]))
		for name, old_value, new_value in zip(field_names, old_fields, new_fields):
			if old_value != new_value:
				print("    {:s}: \"{:s}\" -> \"{:s}\"".format(name, old_value, new_value))

		try:
			answer = input("Update it? [y]es / [N]o / [a]ll the rest / [q]uit asking: ").strip().lower()
		except EOFError:
			answer = 'q'

		if answer.startswith('q'):
			break

		if answer.startswith('a'):
			updates += changed[k:]
			break

		if answer.startswith('y'):
			updates.append((i, old, new))

	return updates



def addNewMeteorites(file_lines, new_lines):
	""" Add to the lines of the existing table the downloaded meteorites that it does not have yet. The
		existing lines are not changed, so manual corrections and entries that are not in the downloaded list
		are kept.

	Arguments:
		file_lines: [list] Lines of the existing table, newest meteorite first.
		new_lines: [list] Lines of the downloaded table.

	Return:
		(merged, added):
			- merged: [list] Lines of the merged table, with the new ones inserted by date.
			- added: [list] Lines that were added.
	"""

	merged = list(file_lines)

	# Names and abbreviations with dates of the meteorites in the table
	names = set()
	idents = set()
	for line in file_lines:
		name, ident = meteoriteIdentity(line)
		names.add(name)
		idents.add(ident)

	added = []
	for line in new_lines:

		name, ident = meteoriteIdentity(line)
		if (name in names) or (ident in idents):
			continue

		names.add(name)
		idents.add(ident)
		added.append(line)

		# Insert the meteorite before the first older one, to keep the most recent fall first
		fields = line.split(';')
		index = len(merged)
		for i, other in enumerate(merged):
			other_fields = other.split(';')
			if (other_fields[4], other_fields[5]) < (fields[4], fields[5]):
				index = i
				break

		merged.insert(index, line)

	return merged, added



def updateMeteoriteOrbits(url="https://www.meteoriteorbits.info/", min_meteorites=40, overwrite=False, \
	ask=None):
	""" Updates wmpl/share/Meteorites_with_Orbits.csv from the list of meteorites with photographic orbits
		kept at meteoriteorbits.info. All the meteorites that the file does not have are added. The ones it
		already has are only changed if you say so: for each one that differs from the list, the differences
		are shown and you are asked whether to update it. The existing file is not touched if the new table
		cannot be read completely.

	Keyword arguments:
		url: [str] Address of the list.
		min_meteorites: [int] Minimum number of meteorites for the download to be accepted. 40 by default.
		overwrite: [bool] Replace the whole file by the downloaded list, losing any manual changes. False by
			default.
		ask: [bool] Ask which of the existing meteorites that differ from the list to update. Without asking,
			they are left as they are. None by default, in which case it asks only when run from a terminal.

	Return:
		[bool] True if the file was updated.
	"""

	socket.setdefaulttimeout(15)

	dir_path = os.path.join(os.path.split(os.path.abspath(__file__))[0], 'wmpl', 'share')
	file_path = os.path.join(dir_path, 'Meteorites_with_Orbits.csv')
	temp_path = file_path + '.part'

	print("Downloading the list of meteorites with orbits from {:s}...".format(url))

	try:
		request = urllibrary.Request(url, headers={'User-Agent': 'Mozilla/5.0 (WesternMeteorPyLib)'})
		html_text = urllibrary.urlopen(request).read().decode('utf-8', 'replace')

	except Exception as e:
		print("Download failed with: " + repr(e))
		return False


	rows = meteoriteTableRows(html_text)
	if len(rows) < 2:
		print("The meteorite table was not found in the page, the existing file is kept.")
		return False


	# Find the columns by their header, so a reordering of the table does not break the update
	header = [cell.lower() for cell in rows[0]]
	cols = [tableColumn(header, key) for key in ('ab', 'name', 'type', 'loc', 'date', 'time', 'zone', 'mass', \
		'a (au)', 'e', 'i (', 'arg', 'asc', 'ref')]

	if None in cols:
		print("Unexpected table header {:s}, the existing file is kept.".format(repr(rows[0])))
		return False


	# Convert the rows of the table to lines of the file
	lines = []
	for row in rows[1:]:

		# Keep the existing file if a row was not read completely
		if len(row) < len(header):
			print("Incomplete row {:s}, the existing file is kept.".format(repr(row)))
			return False

		ab, name, mtype, country, date, time, zone, mass = [row[c] for c in cols[:8]]
		ref = row[cols[13]]

		# Remove the footnote marks and the typographic apostrophes, as in the existing file
		name = name.rstrip('*').replace(u'\u2019', "'")
		mtype = mtype.rstrip('*')

		# Write an unknown mass, given as a dash, as "???" and keep the hyphen of a mass range
		mass_parts = []
		for part in mass.split('/'):
			if part.strip() == u'\u2013':
				mass_parts.append(part.replace(u'\u2013', '???'))
			else:
				mass_parts.append(part.replace(u'\u2013', '-'))

		mass = '/'.join(mass_parts)

		# Read the orbital elements and their errors
		fields = [ab, name, mtype, country, date.replace('/', '-'), time, zone, mass]
		for c in cols[8:13]:

			value, err = valueError(row[c])
			if value is None:
				print("Cannot read \"{:s}\" for {:s}, the existing file is kept.".format(row[c], name))
				return False

			fields += [value, err]

		# The semi-major axis and the eccentricity are required by loadMeteoritesElements()
		if (fields[8] == '???') or (fields[10] == '???'):
			print("No a or e for {:s}, the existing file is kept.".format(name))
			return False

		fields.append(ref)
		lines.append(';'.join(fields))


	if len(lines) < min_meteorites:
		print("Only {:d} meteorites were read, the existing file is kept.".format(len(lines)))
		return False

	n_downloaded = len(lines)


	# Read the date of the list, e.g. "As of September 22nd, 2026"
	date_match = re.search(r'As of\s+([A-Za-z]+)\s+(\d+)\s*(?:st|nd|rd|th)?\s*,\s*(\d{4})', \
		re.sub(r'<[^>]+>', ' ', html_text))
	try:
		list_date = datetime.datetime.strptime("{:s} {:s} {:s}".format(*date_match.groups()), '%B %d %Y')

	except Exception:
		list_date = datetime.datetime.utcnow()


	# Add the new meteorites to the existing file, and update the existing ones only if asked to
	if (not overwrite) and os.path.isfile(file_path):

		with open(file_path, 'r', encoding='utf-8', newline='') as f:
			file_lines = [line.rstrip('\r\n') for line in f if line.strip() and not line.startswith('#')]

		if ask is None:
			ask = sys.stdin.isatty()

		# Ask which of the existing meteorites to update
		updated = []
		changed = changedMeteorites(file_lines, lines)
		if changed:
			if ask:
				updated = askUpdates(changed)
			else:
				print("{:d} meteorites in the file differ from the list and are left as they are.".format(\
					len(changed)))

		for i, old, new in updated:
			file_lines[i] = new

		lines, added = addNewMeteorites(file_lines, lines)

		if (not added) and (not updated):
			print(" ... done! Nothing was added or updated among the {:d} meteorites of the list, the file is " \
				"unchanged.".format(n_downloaded))
			return False

		if added:
			print("Added {:d} new meteorites: {:s}".format(len(added), \
				", ".join(line.split(';')[1] for line in added)))

		if updated:
			print("Updated {:d} meteorites: {:s}".format(len(updated), \
				", ".join(new.split(';')[1] for _, _, new in updated)))


	# Write the table to a temporary file first, so a failed write does not destroy the existing one
	with open(temp_path, 'w', encoding='utf-8', newline='') as f:
		f.write("# Updated on: {:s}".format(list_date.strftime('%Y/%m/%d')) + ';'*18 + '\n')
		f.write(METEORITES_HEADER + '\n')
		for line in lines:
			f.write(line + '\n')

	shutil.move(temp_path, file_path)

	print(" ... done! {:d} meteorites with orbits, list of {:s}.".format(len(lines), \
		list_date.strftime('%Y-%m-%d')))

	return True




if __name__ == "__main__":

	import argparse


	### COMMAND LINE ARGUMENTS

	# Init the command line arguments parser
	arg_parser = argparse.ArgumentParser(description="Updates the asteroid and comet orbit files, or with --meteorites the table of meteorites with orbits.")

	arg_parser.add_argument('--meteorites', action='store_true', \
		help="Instead of the asteroid and comet files, check the table of meteorites with orbits (wmpl/share/Meteorites_with_Orbits.csv) against meteoriteorbits.info. All the new meteorites are added, and for the existing ones that differ you are asked, one by one, whether to update them. This is never done automatically, to keep the manual corrections of the table.")

	# Parse the command line arguments
	cml_args = arg_parser.parse_args()

	if cml_args.meteorites:
		updateMeteoriteOrbits()

	else:
		updateOrbitFiles()
