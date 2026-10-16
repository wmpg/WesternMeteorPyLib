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

else:
	import urllib.request as urllibrary

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
METEORITES_HEADER = "# Ab.;Name;Type;Country;Date (UTC);Time (UTC);Time Zone;Mass (kg);a (AU);a error (AU);e;" \
	"e error (AU);i (°);i error (°);Arg. Peri. (°);Arg. Peri. Error (°);Asc. node (°);Asc. node error (°);Ref."


def _meteoriteTableRows(html_text):
	""" Extract the rows of the meteorite table from the HTML of meteoriteorbits.info.

	Arguments:
		html_text: [str] HTML of the page.

	Return:
		[list] Rows of the table, each a list of cell strings. The first row is the header.
	"""

	if sys.version_info.major < 3:
		from HTMLParser import HTMLParser
	else:
		from html.parser import HTMLParser


	class TableParser(HTMLParser):

		def __init__(self):
			HTMLParser.__init__(self)
			self.tables = []
			self.rows = None
			self.row = None
			self.cell = None

		def handle_starttag(self, tag, attrs):
			if tag == 'table':
				self.rows = []
			elif (tag == 'tr') and (self.rows is not None):
				self.row = []
			elif (tag in ('td', 'th')) and (self.row is not None):
				self.cell = ''
			elif (tag == 'br') and (self.cell is not None):
				self.cell += ' '

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


	parser = TableParser()
	parser.feed(html_text)

	# The meteorite table is the one with the orbital elements in its header
	for rows in parser.tables:
		if rows and any('a (AU)' in cell for cell in rows[0]):
			return rows

	return []


def _valueError(cell):
	""" Split a cell such as "1.844 ±0.003" into its value and error. Returns ("???", "") for an unknown value, and
		(None, None) if the cell cannot be read.
	"""

	cell = cell.replace(u'\u2212', '-').replace(u'\u2013', '').replace('*', '').strip()

	if (not cell) or (cell == '???'):
		return '???', ''

	# Symmetric error, e.g. "1.844 ±0.003"; an error of zero means that none was given
	match = re.match(u'^([-+]?\\d+(?:\\.\\d+)?)\\s*(?:\u00b1\\s*(\\d+(?:\\.\\d+)?))?$', cell)
	if match:
		err = match.group(2) or ''
		if err and (float(err) == 0):
			err = ''
		return match.group(1), err

	# Asymmetric error, e.g. "2.62 +0.7-0.46"
	match = re.match(r'^([-+]?\d+(?:\.\d+)?)\s*\+(\d+(?:\.\d+)?)\s*-(\d+(?:\.\d+)?)$', cell)
	if match:
		return match.group(1), '+{:s}/-{:s}'.format(match.group(2), match.group(3))

	return None, None


def _meteoriteIdentity(line):
	""" Identify a line of the meteorite table. Return the name without accents, case and punctuation, and the
		abbreviation with the date; either one matching means it is the same meteorite (the second catches names
		that were corrected or are spelled differently).
	"""

	fields = line.split(';')

	name = unicodedata.normalize('NFKD', fields[1])
	name = ''.join(c.lower() for c in name if c.isalnum() and not unicodedata.combining(c))

	return name, (fields[0], fields[4])


def _changedMeteorites(file_lines, new_lines):
	""" Find the meteorites of the existing table whose downloaded line is not the same.

	Arguments:
		file_lines: [list] Lines of the existing table.
		new_lines: [list] Lines of the downloaded table.

	Return:
		[list] Tuples (index in file_lines, existing line, downloaded line).
	"""

	by_name = {}
	by_ident = {}
	for i, line in enumerate(file_lines):
		name, ident = _meteoriteIdentity(line)
		by_name[name] = i
		by_ident[ident] = i

	changed = []
	for line in new_lines:

		name, ident = _meteoriteIdentity(line)
		i = by_name.get(name, by_ident.get(ident))

		if (i is not None) and (file_lines[i] != line):
			changed.append((i, file_lines[i], line))

	return changed


def _askUpdates(changed):
	""" Show how the downloaded values differ from the existing ones and ask which meteorites to update. The
		default answer is to keep the existing line.

	Arguments:
		changed: [list] Tuples (index, existing line, downloaded line) from _changedMeteorites().

	Return:
		[list] The tuples of the meteorites to update.
	"""

	field_names = [name.lstrip('# ') for name in METEORITES_HEADER.split(';')]

	print()
	print('{:d} meteorites in the file differ from the list. The file is kept unless you say yes.'.format(
		len(changed)))

	updates = []
	for k, (i, old, new) in enumerate(changed):

		old_fields = old.split(';')
		new_fields = new.split(';')

		print()
		print('[{:d}/{:d}] {:s} ({:s})'.format(k + 1, len(changed), old_fields[1], old_fields[0]))
		for name, old_value, new_value in zip(field_names, old_fields, new_fields):
			if old_value != new_value:
				print('    {:s}: "{:s}" -> "{:s}"'.format(name, old_value, new_value))

		try:
			answer = input('Update it? [y]es / [N]o / [a]ll the rest / [q]uit asking: ').strip().lower()
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


def _addNewMeteorites(file_lines, new_lines):
	""" Add to the lines of the existing table the downloaded meteorites that it does not have yet. The existing lines
		are not changed, so manual corrections and entries that are not in the downloaded list survive.

	Arguments:
		file_lines: [list] Lines of the existing table, newest meteorite first.
		new_lines: [list] Lines of the downloaded table.

	Return:
		[list] The lines of the merged table (the new ones inserted by date), [list] the lines that were added.
	"""

	merged = list(file_lines)

	names = set()
	ids = set()
	for line in file_lines:
		name, ident = _meteoriteIdentity(line)
		names.add(name)
		ids.add(ident)

	added = []
	for line in new_lines:

		name, ident = _meteoriteIdentity(line)
		if (name in names) or (ident in ids):
			continue

		names.add(name)
		ids.add(ident)
		added.append(line)

		# Keep the table sorted with the most recent fall first, as in the source
		fields = line.split(';')
		index = len(merged)
		for i, other in enumerate(merged):
			other_fields = other.split(';')
			if (other_fields[4], other_fields[5]) < (fields[4], fields[5]):
				index = i
				break

		merged.insert(index, line)

	return merged, added


def updateMeteoriteOrbits(url="https://www.meteoriteorbits.info/", min_meteorites=40, overwrite=False, ask=None):
	""" Updates wmpl/share/Meteorites_with_Orbits.csv from the list of meteorites with photographic orbits kept at
		meteoriteorbits.info. All the meteorites that the file does not have are added. The ones it already has are
		never changed unless you say so: for each one that differs from the list, the differences are shown and you
		are asked whether to update it, so manual corrections and entries that are not in the list are kept. The
		existing file is not touched if the new table is not read completely; the reason is printed.

	Keyword arguments:
		url: [str] Address of the list.
		min_meteorites: [int] Minimum number of meteorites for the download to be accepted.
		overwrite: [bool] Replace the whole file by the downloaded list, losing any manual changes.
		ask: [bool] Ask which of the existing meteorites that differ from the list to update. None (default) asks
			only when run from a terminal; without asking the existing meteorites are left as they are.

	Return:
		[bool] True if the file was updated.
	"""

	socket.setdefaulttimeout(15)

	dir_path = os.path.join(os.path.split(os.path.abspath(__file__))[0], 'wmpl', 'share')
	file_path = os.path.join(dir_path, 'Meteorites_with_Orbits.csv')
	temp_path = file_path + '.part'

	print('Downloading the list of meteorites with orbits from {:s}...'.format(url))

	try:
		request = urllibrary.Request(url, headers={'User-Agent': 'Mozilla/5.0 (WesternMeteorPyLib)'})
		html_text = urllibrary.urlopen(request).read().decode('utf-8', 'replace')
	except Exception as e:
		print('Download failed with: ' + repr(e))
		return False

	rows = _meteoriteTableRows(html_text)
	if len(rows) < 2:
		print('The meteorite table was not found in the page, the existing file is kept.')
		return False

	# Map the columns by their header, so a reordering of the table does not break the update
	header = [cell.lower() for cell in rows[0]]

	def column(*keys):
		for i, name in enumerate(header):
			if any(name.startswith(key) for key in keys):
				return i
		raise KeyError(keys)

	try:
		cols = [column(key) for key in ('ab', 'name', 'type', 'loc', 'date', 'time', 'zone', 'mass', 'a (au)', 'e',
			'i (', 'arg', 'asc', 'ref')]
	except KeyError as e:
		print('Unexpected table header {:s}, the existing file is kept.'.format(repr(rows[0])))
		return False

	lines = []
	for row in rows[1:]:

		if len(row) < len(header):
			continue

		ab, name, mtype, country, date, time, zone, mass = [row[c] for c in cols[:8]]
		ref = row[cols[13]]

		# Footnote marks and typographic apostrophes, as in the existing file
		name = name.rstrip('*').replace(u'\u2019', "'")
		mtype = mtype.rstrip('*')

		# An unknown mass is a dash; a mass range keeps its hyphen
		mass = '/'.join([part.replace(u'\u2013', '???') if part.strip() == u'\u2013' else part.replace(u'\u2013', '-')
			for part in mass.split('/')])

		fields = [ab, name, mtype, country, date.replace('/', '-'), time, zone, mass]
		for c in cols[8:13]:
			value, err = _valueError(row[c])
			if value is None:
				print('Cannot read "{:s}" for {:s}, the existing file is kept.'.format(row[c], name))
				return False
			fields += [value, err]

		# The semi-major axis and the eccentricity are required by loadMeteoritesElements()
		if (fields[8] == '???') or (fields[10] == '???'):
			print('No a or e for {:s}, the existing file is kept.'.format(name))
			return False

		fields.append(ref)
		lines.append(';'.join(fields))

	if len(lines) < min_meteorites:
		print('Only {:d} meteorites were read, the existing file is kept.'.format(len(lines)))
		return False

	n_downloaded = len(lines)

	# Date of the list, e.g. "As of September 22nd, 2026"
	date_match = re.search(r'As of\s+([A-Za-z]+)\s+(\d+)\s*(?:st|nd|rd|th)?\s*,\s*(\d{4})',
		re.sub(r'<[^>]+>', ' ', html_text))
	try:
		list_date = datetime.datetime.strptime('{:s} {:s} {:s}'.format(*date_match.groups()), '%B %d %Y')
	except Exception:
		list_date = datetime.datetime.utcnow()

	# Add the new meteorites to the existing file, and update the existing ones only if asked to
	if (not overwrite) and os.path.isfile(file_path):

		with open(file_path, 'r', encoding='utf-8', newline='') as f:
			file_lines = [line.rstrip('\r\n') for line in f if line.strip() and not line.startswith('#')]

		if ask is None:
			ask = sys.stdin.isatty()

		updated = []
		changed = _changedMeteorites(file_lines, lines)
		if changed:
			if ask:
				updated = _askUpdates(changed)
			else:
				print('{:d} meteorites in the file differ from the list and are left as they are.'.format(
					len(changed)))

		for i, old, new in updated:
			file_lines[i] = new

		lines, added = _addNewMeteorites(file_lines, lines)

		if (not added) and (not updated):
			print(' ... done! Nothing was added or updated among the {:d} meteorites of the list, the file is '
				'unchanged.'.format(n_downloaded))
			return False

		if added:
			print('Added {:d} new meteorites: {:s}'.format(len(added), ', '.join(l.split(';')[1] for l in added)))
		if updated:
			print('Updated {:d} meteorites: {:s}'.format(len(updated), ', '.join(l[2].split(';')[1] for l in updated)))

	with open(temp_path, 'w', encoding='utf-8', newline='') as f:
		f.write('# Updated on: {:s}'.format(list_date.strftime('%Y/%m/%d')) + ';'*18 + '\n')
		f.write(METEORITES_HEADER + '\n')
		for line in lines:
			f.write(line + '\n')

	shutil.move(temp_path, file_path)

	print(' ... done! {:d} meteorites with orbits, list of {:s}.'.format(len(lines), list_date.strftime('%Y-%m-%d')))

	return True




if __name__ == "__main__":

	import argparse

	arg_parser = argparse.ArgumentParser(description="Updates the asteroid and comet orbit files, or with --meteorites "
		"the table of meteorites with orbits.")

	arg_parser.add_argument('--meteorites', action='store_true', help="""Instead of the asteroid and comet files,
		check the table of meteorites with orbits (wmpl/share/Meteorites_with_Orbits.csv) against meteoriteorbits.info:
		all the new meteorites are added, and for the existing ones that differ you are asked, one by one, whether to
		update them. This is never done automatically, to keep the manual corrections of the table.""")

	cml_args = arg_parser.parse_args()

	if cml_args.meteorites:
		updateMeteoriteOrbits()
	else:
		updateOrbitFiles()