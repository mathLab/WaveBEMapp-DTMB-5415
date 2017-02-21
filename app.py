"""
Utilities for handling the Graphic Unit Interface.
"""
# try: import tkinter
# except ImportError:
# 	import Tkinter as tkinter
# 	import ttk
# 	#from tkFileDialog import askopenfilename
# else:
# 	from tkinter import ttk
# 	#from tkinter.filedialog import askopenfilename

from Tkinter import *
from PIL import Image, ImageTk
import logging

def write_prm(foutput, X):
	"""
	to doc
	"""
	with open(foutput, 'w') as output_file:
		output_file.write('\n[Box info]\n')
		output_file.write('# This section collects all the properties of the FFD bounding box.\n')

		output_file.write('\n# n control points indicates the number of control points in each direction (x, y, z).\n')
		output_file.write('# For example, to create a 2 x 3 x 2 grid, use the following: n control points: 2, 3, 2\n')
		output_file.write('n control points x: 2\n')
		output_file.write('n control points y: 2\n')
		output_file.write('n control points z: 2\n')

		output_file.write('\n# box lenght indicates the length of the FFD bounding box along the three canonical directions (x, y, z).\n')
		output_file.write('# It uses the local coordinate system.\n')
		output_file.write('# For example to create a 2 x 1.5 x 3 meters box use the following: lenght box: 2.0, 1.5, 3.0\n')
		output_file.write('box lenght x: 4400.0\n')
		output_file.write('box lenght y: 460.0\n')
		output_file.write('box lenght z: 480.0\n')

		output_file.write('\n# box origin indicates the x, y, and z coordinates of the origin of the FFD bounding box. That is center of\n')
		output_file.write('# rotation of the bounding box. It corresponds to the point coordinates with position [0][0][0].\n')
		output_file.write('# See section "Parameters weights" for more details.\n')
		output_file.write('# For example, if the origin is equal to 0., 0., 0., use the following: origin box: 0., 0., 0.\n')
		output_file.write('box origin x: -1200.0\n')
		output_file.write('box origin y: 0.0\n')
		output_file.write('box origin z: -250.0\n')

		output_file.write('\n# rotation angle indicates the rotation angle around the x, y, and z axis of the FFD bounding box in degrees.\n')
		output_file.write('# The rotation is done with respect to the box origin.\n')
		output_file.write('# For example, to rotate the box by 2 deg along the z direction, use the following: rotation angle: 0., 0., 2.\n')
		output_file.write('rotation angle x: 0\n')
		output_file.write('rotation angle y: 0\n')
		output_file.write('rotation angle z: 0\n')

		output_file.write('\n\n[Parameters weights]\n')
		output_file.write('# This section describes the weights of the FFD control points.\n')
		output_file.write('# We adopt the following convention:\n')
		output_file.write('# For example with a 2x2x2 grid of control points we have to fill a 2x2x2 matrix of weights.\n')
		output_file.write('# If a weight is equal to zero you can discard the line since the default is zero.\n')
		output_file.write('#\n')
		output_file.write('# | x index | y index | z index | weight |\n')
		output_file.write('#  --------------------------------------\n')
		output_file.write('# |    0    |    0    |    0    |  1.0   |\n')
		output_file.write('# |    0    |    1    |    1    |  0.0   | --> you can erase this line without effects\n')
		output_file.write('# |    0    |    1    |    0    | -2.1   |\n')
		output_file.write('# |    0    |    0    |    1    |  3.4   |\n')
			
		output_file.write('\n# parameter x collects the displacements along x, normalized with the box lenght x.')
		output_file.write('\nparameter x: 0   0   0   0.0\n')
		output_file.write(13 * ' ' + '0   0   1   0.0\n')
		output_file.write(13 * ' ' + '0   1   0   0.0\n')
		output_file.write(13 * ' ' + '0   1   1   0.0\n')
		output_file.write(13 * ' ' + '1   0   0   0.0\n')
		output_file.write(13 * ' ' + '1   0   1   0.0\n')
		output_file.write(13 * ' ' + '1   1   0   0.0\n')
		output_file.write(13 * ' ' + '1   1   1   0.0\n')

		output_file.write('\n# parameter y collects the displacements along y, normalized with the box lenght y.')
		output_file.write('\nparameter y: 0   0   0   0.0\n')
		output_file.write(13 * ' ' + '0   0   1   0.0\n')
		output_file.write(13 * ' ' + '0   1   0   ' + str(X[2]) + '\n')
		output_file.write(13 * ' ' + '0   1   1   ' + str(X[0]) + '\n')
		output_file.write(13 * ' ' + '1   0   0   0.0\n')
		output_file.write(13 * ' ' + '1   0   1   0.0\n')
		output_file.write(13 * ' ' + '1   1   0   ' + str(X[3]) + '\n')
		output_file.write(13 * ' ' + '1   1   1   ' + str(X[1]) + '\n')
	
		output_file.write('\n# parameter z collects the displacements along z, normalized with the box lenght z.')
		output_file.write('\nparameter z: 0   0   0   0.0\n')
		output_file.write(13 * ' ' + '0   0   1   0.0\n')
		output_file.write(13 * ' ' + '0   1   0   ' + str(X[4]) + '\n')
		output_file.write(13 * ' ' + '0   1   1   0.0\n')
		output_file.write(13 * ' ' + '1   0   0   0.0\n')
		output_file.write(13 * ' ' + '1   0   1   0.0\n')
		output_file.write(13 * ' ' + '1   1   0   ' + str(X[5]) + '\n')
		output_file.write(13 * ' ' + '1   1   1   0.0\n')


def read_input_output(finput, dF=False):
	"""Reads a dat file called finput. Returns the inputs and the 
	corresponding outputs. If dF is True it also reads the gradients.
	"""
	import numpy as np
	with open(finput, 'r') as input_file:
		for line in input_file:
			M, m = line.split()
			M = int(M)
			m = int(m)
			break

	X = np.zeros((M, m))
	F = np.zeros((M, 1))
	if dF is True:
		dFX = np.zeros((M, m))

	i = -1
	for line in open(finput, 'r'):
		columns = line.split()
		if i >= 0:
			for j in range(m):
				X[i,j] = columns[j]
			F[i] = columns[m]
			if dF is True:
				for k in range(m):
					dFX[i,k] = columns[m+1+k]
		i += 1

	if dF is True:
		return (X, F, dFX)
	else:
		return (X, F)


def generate_png(shape):
	"""
	Draw a TopoDS_Shape with matplotlib
	"""
	from OCC.Visualization import Tesselator
	from mpl_toolkits.mplot3d import Axes3D
	from matplotlib import pyplot as plt
	from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
	
	tess = Tesselator(shape)
	triangles = []
	edges = []

	# get the triangles
	triangle_count = tess.ObjGetTriangleCount()
	for i_triangle in range(0, triangle_count):
		i1, i2, i3 = tess.GetTriangleIndex(i_triangle)
		triangles.append([tess.GetVertex(i1), tess.GetVertex(i2), tess.GetVertex(i3)])

	# get the edges
	edge_count = tess.ObjGetEdgeCount()
	for i_edge in range(0, edge_count):
		vertex_count = tess.ObjEdgeGetVertexCount(i_edge)
		edge = []
		for i_vertex in range(0, vertex_count):
			vertex = tess.GetEdgeVertex(i_edge, i_vertex)
			edge.append(vertex)
		edges.append(edge)

	# plot it
	fig_side = plt.figure(figsize=(10, 4))
	ax = Axes3D(fig_side)

	ax.add_collection3d(Poly3DCollection(triangles, linewidths=0.15, alpha=0.5))
	ax.add_collection3d(Line3DCollection(edges, colors='w', linewidths=1.0))

	ax.set_axis_off()

	ax.set_xlim(-1800, 1800)
	ax.set_ylim(-800, 800)
	ax.set_zlim(-800, 800)
	ax.view_init(elev=-1., azim=90)
	fig_side.savefig("views/side.png")

	
	fig_top = plt.figure(figsize=(5, 4))
	ax_top = Axes3D(fig_top)

	ax_top.add_collection3d(Poly3DCollection(triangles, linewidths=0.15, alpha=0.5))
	ax_top.add_collection3d(Line3DCollection(edges, colors='w', linewidths=1.0))

	ax_top.set_axis_off()

	ax_top.set_xlim(-2500, 2500)
	ax_top.set_ylim(-50, 450)
	ax_top.set_zlim(-250, 250)
	ax_top.view_init(elev=-1., azim=-2)
	fig_top.savefig("views/back.png")

	ax_top.set_xlim(-2500, 2500)
	ax_top.set_ylim(-50, 450)
	ax_top.set_zlim(-200, 300)
	ax_top.view_init(elev=-1., azim=182)
	fig_top.savefig("views/front.png")


class Window(Tk):
	def __init__(self, parent):
		Tk.__init__(self, parent)
		self.parent = parent
		self.prediction = None
		self.precompute_active_subspace()
		self.initialize()

	def initialize(self):
		self.geometry("800x300+30+30")
		
		#######################################################################################
		# Frame parametri
		#######################################################################################
		params_frame = Frame(self, width=400, height=300)
		params_frame.pack(side="left", padx=5, pady=5, fill=BOTH, expand=True)
		params_frame.pack_propagate(0)
		Label(params_frame, text="Parameters", font=("Arial", 20), padx=50, anchor='w').pack(side='top')

		#######################################################################################
		# Frame 1 con i primi 3
		#######################################################################################
		frame1 = Frame(params_frame, width=200, height=75)
		frame1.pack(side="top", padx=5, pady=5, fill=BOTH, expand=True)
		frame1.pack_propagate(0)
		
		self.var1 = DoubleVar()
		self.w1 = Scale(frame1, from_=-0.2, to=0.3, orient=HORIZONTAL, variable=self.var1,
					resolution=0.001, sliderlength=25, length=105, label='Control point 1')
		self.w1.set(0.00)
		self.w1.bind('<ButtonRelease-1>', self.compute_solution)
		self.w1.pack(side='left', padx=10)
		
		self.var2 = DoubleVar()
		self.w2 = Scale(frame1, from_=-0.2, to=0.3, orient=HORIZONTAL, variable=self.var2,
					resolution=0.001, sliderlength=25, length=105, label='Control point 2')
		self.w2.set(0.00)
		self.w2.bind('<ButtonRelease-1>', self.compute_solution)
		self.w2.pack(side='left', padx=10)
		
		self.var3 = DoubleVar()
		self.w3 = Scale(frame1, from_=-0.2, to=0.3, orient=HORIZONTAL, variable=self.var3,
					resolution=0.001, sliderlength=25, length=105, label='Control point 3')
		self.w3.set(0.00)
		self.w3.bind('<ButtonRelease-1>', self.compute_solution)
		self.w3.pack(side='left', padx=10)

		#######################################################################################
		# Frame 2 con i secondi 3
		#######################################################################################
		frame2 = Frame(params_frame, width=200, height=75)
		frame2.pack(side="top", padx=5, pady=5, fill=BOTH, expand=True)
		frame2.pack_propagate(0)
		
		self.var4 = DoubleVar()
		self.w4 = Scale(frame2, from_=-0.2, to=0.3, orient=HORIZONTAL, variable=self.var4,
					resolution=0.001, sliderlength=25, length=105, label='Control point 4')
		self.w4.set(0.00)
		self.w4.bind('<ButtonRelease-1>', self.compute_solution)
		self.w4.pack(side='left', padx=10)
		
		self.var5 = DoubleVar()
		self.w5 = Scale(frame2, from_=-0.2, to=0.5, orient=HORIZONTAL, variable=self.var5,
					resolution=0.001, sliderlength=25, length=105, label='Control point 5')
		self.w5.set(0.00)
		self.w5.bind('<ButtonRelease-1>', self.compute_solution)
		self.w5.pack(side='left', padx=10)
		
		self.var6 = DoubleVar()
		self.w6 = Scale(frame2, from_=-0.2, to=0.5, orient=HORIZONTAL, variable=self.var6,
					resolution=0.001, sliderlength=25, length=105, label='Control point 6')
		self.w6.set(0.00)
		self.w6.bind('<ButtonRelease-1>', self.compute_solution)
		self.w6.pack(side='left', padx=10)		
		
		#######################################################################################
		# Frame 3 con gli ultimi 2
		#######################################################################################
		frame3 = Frame(params_frame, width=200, height=75)
		frame3.pack(side="top", padx=5, pady=5, fill=BOTH, expand=True)
		frame3.pack_propagate(0)
		
		self.var7 = IntVar()
		self.w7 = Scale(frame3, from_=500, to=800, orient=HORIZONTAL, variable=self.var7,
					resolution=1, sliderlength=25, length=105, label='Weight')
		self.w7.set(650)
		self.w7.bind('<ButtonRelease-1>', self.compute_solution)
		self.w7.pack(side='left', padx=10)
		
		self.var8 = DoubleVar()
		self.w8 = Scale(frame3, from_=0.250, to=0.360, orient=HORIZONTAL, variable=self.var8,
					resolution=0.001, sliderlength=25, length=105, label='Froude')
		self.w8.set(0.3)
		self.w8.bind('<ButtonRelease-1>', self.compute_solution)
		self.w8.pack(side='left', padx=10)

		#######################################################################################
		# Frame soluzione
		#######################################################################################
		sol_frame = Frame(self, width=400, height=300)
		sol_frame.pack(side="left", padx=5, pady=5, fill=BOTH, expand=True)
		sol_frame.pack_propagate(0)
		
		Label(sol_frame, text="Solution", font=("Arial", 20), padx=50, anchor='w').pack(side='top')
		
		#######################################################################################
		# Frame valori
		#######################################################################################
		frame4 = Frame(sol_frame)#, width=200, height=250)
		frame4.pack(side="top", padx=5, pady=5, fill=BOTH, expand=True)
		frame4.pack_propagate(0)
		
		self.label = Label(frame4, pady=10)
		self.label.pack()
		self.update_label()
		
		self.show_Button = Button(frame4, text='Show Hull', command=self.OnShowClick)
		self.show_Button.pack()

		self.close_Button = Button(frame4, text='Close', command=self.OnCloseClick)
		self.close_Button.pack()
		

	def update_values(self):
		self.control1 = self.var1.get()
		self.control2 = self.var2.get()
		self.control3 = self.var3.get()
		self.control4 = self.var4.get()
		self.control5 = self.var5.get()
		self.control6 = self.var6.get()
		self.weight = self.var7.get()
		self.velocity = self.var8.get()*1000


	def update_label(self):
		self.update_values()
		selection = "Control point 1: " + str(self.control1) + '\n' + \
			"Control point 2: " + str(self.control2) + '\n' + \
			"Control point 3: " + str(self.control3) + '\n' + \
			"Control point 4: " + str(self.control4) + '\n' + \
			"Control point 5: " + str(self.control5) + '\n' + \
			"Control point 6: " + str(self.control6) + '\n' + \
			"Weight: " + str(self.weight) + '\n' + \
			"Froude: " + str(self.velocity/1000) + '\n\n' + \
			"Wave resistance: " + str(self.prediction)
		self.label.config(text=selection)


	def precompute_active_subspace(self):
		import active_subspaces as ac
		import numpy as np
		import pygem.igeshandler as ih

		X1, fX1 = read_input_output("input_output_total", dF=False)
		self.ss = ac.subspaces.Subspaces()

		#Estimated gradients using local linear models
		dfX0 = ac.gradients.local_linear_gradients(X1, fX1)#, p=15) 

		self.ss.compute(df=dfX0, nboot=1000)
		self.ss.partition(2)

		RS = ac.utils.response_surfaces.PolynomialApproximation(3)
		#Train the surface with active variable values (y = XX.dot(self.ss.W1)) and function values (f)
		y1 = X1.dot(self.ss.W1)
		RS.train(y1, fX1)

		avdom = ac.domains.BoundedActiveVariableDomain(self.ss)
		avmap = ac.domains.BoundedActiveVariableMap(avdom)
		self.asrs = ac.response_surfaces.ActiveSubspaceResponseSurface(avmap=avmap, respsurf=RS)

		# Since we have to load only one time the original mesh we put it here
		self.original_mesh_points = np.load('goteborg_original.npy')

		self.handler = ih.IgesHandler()
		self.handler.infile = 'goteborg_original.iges'
		self.handler._control_point_position = np.load('control_position.npy').tolist()
		self.handler.shape = self.handler.load_shape_from_file('goteborg_original.iges')



	def compute_solution(self, event):
		import numpy as np
		self.update_values()

		new_input = np.array([self.control1, self.control2, self.control3, self.control4,
			self.control5, self.control6, self.weight, self.velocity])
		X_LOW = np.array([-0.2, -0.2, -0.2, -0.2, -0.2, -0.2, 500, 250])
		X_UP = np.array([0.3, 0.3, 0.3, 0.3, 0.5, 0.5, 800, 360])
		# rescale in the simmetric space
		D = np.diag(X_UP - X_LOW)
		new_input_rescaled = np.linalg.inv(D) * (2 * new_input.transpose() - (X_UP + X_LOW))

		self.prediction = self.asrs.predict(np.array([new_input_rescaled.diagonal()]))[0][0][0]

		self.update_label()


	def deform_hull(self):
		import pygem as pg
		import pygem.freeform as ffd
		import numpy as np

		new_input = np.array([self.control1, self.control2, self.control3, self.control4,
			self.control5, self.control6, self.weight, self.velocity])
		write_prm('parameters.prm', new_input)
		params = pg.params.FFDParameters()
		params.read_parameters(filename='parameters.prm')
		
		free_form = ffd.FFD(params, self.original_mesh_points)
		free_form.perform()

		new_mesh_points = free_form.modified_mesh_points
		self.handler.write(new_mesh_points, 'goteborg_new.iges')

		# import time
		# t1 = time.time()
		# t2 = time.time()
		# print 'Total: {:.6f} s'.format(t2 - t1)

	def OnShowClick(self):
		self.deform_hull()
		hull = self.handler.load_shape_from_file('goteborg_new.iges')
		generate_png(hull)
		self.create_figs_window()

		# from OCC.Display.WebGl import threejs_renderer
		
		# web_renderer = threejs_renderer.ThreejsRenderer()
		# web_renderer.DisplayShape(hull)
		# web_renderer.render()


	def create_figs_window(self):
		self.top = Toplevel()
		self.top.title("Deformed hull")

		# pick an image file you have .bmp  .jpg  .gif.  .png
		# load the file and covert it to a Tkinter image object
		front_img = Image.open("views/front.png")
		[front_img_width, front_img_height] = front_img.size

		back_img = Image.open("views/back.png")
		[back_img_width, back_img_height] = back_img.size

		side_img = Image.open("views/side.png")
		[side_img_width, side_img_height] = side_img.size

		same = False
		scaling_factor = 1.2
		if same:
			new_front_img_width = int(front_img_width)
			new_front_img_height = int(front_img_height) 
			new_back_img_width = int(back_img_width)
			new_back_img_height = int(back_img_height) 
		else:
			new_front_img_width = int(front_img_width/scaling_factor)
			new_front_img_height = int(front_img_height/scaling_factor) 
			new_back_img_width = int(back_img_width/scaling_factor)
			new_back_img_height = int(back_img_height/scaling_factor)

		new_side_img_width = int(side_img_width)
		new_side_img_height = int(side_img_height)
 
		front_img = front_img.resize((new_front_img_width, new_front_img_height), Image.ANTIALIAS)
		back_img = back_img.resize((new_back_img_width, new_back_img_height), Image.ANTIALIAS)
		side_img = side_img.resize((new_side_img_width, new_side_img_height), Image.ANTIALIAS)	

		self.image1 = ImageTk.PhotoImage(front_img)
		self.image2 = ImageTk.PhotoImage(back_img)
		self.image3 = ImageTk.PhotoImage(side_img)

		# make the root window the size of the image
		# 0, 0: position coordinates of root 'upper left corner'
		self.top.geometry("%dx%d+%d+%d" % (new_side_img_width + 150,
			new_front_img_height + new_side_img_height + 80, 0, 0))
		self.top.transient(self)


		#######################################################################################
		# Frame riga di testo 1 - front and back
		#######################################################################################
		text_frame = Frame(self.top, height=25)
		text_frame.pack(side='top', padx=5, pady=5, fill=BOTH, expand=False)
		text_frame.pack_propagate(0)
		Label(text_frame, text="Front", font=("Arial", 20), anchor='n').pack(side='left', fill=BOTH, expand=True)
		Label(text_frame, text="Back", font=("Arial", 20), anchor='n').pack(side='left', fill=BOTH, expand=True)

		#######################################################################################
		# Frame figura 1 - front and back
		#######################################################################################
		self.geo_frame = Frame(self.top)
		self.geo_frame.pack(side='top', padx=5, pady=5, fill=BOTH, expand=True)
		self.geo_frame.pack_propagate(0)

		# root has no image argument, so use a label as a panel
		self.panel1 = Label(self.geo_frame, image=self.image1, padx=5)
		self.panel1.pack(side='left', fill=BOTH, expand=True)

		self.panel2 = Label(self.geo_frame, image=self.image2, padx=5)
		self.panel2.pack(side='right', fill=BOTH, expand=True)


		#######################################################################################
		# Frame riga di testo 2 - side
		#######################################################################################
		text_frame2 = Frame(self.top, height=25)
		text_frame2.pack(side='top', padx=5, pady=5, fill=BOTH, expand=False)
		text_frame2.pack_propagate(0)
		Label(text_frame2, text="Side", font=("Arial", 20), anchor='n').pack(side='left', fill=BOTH, expand=True)

		#######################################################################################
		# Frame figura 2 - side
		#######################################################################################
		self.geo_frame2 = Frame(self.top)
		self.geo_frame2.pack(side='top', padx=5, pady=5, fill=BOTH, expand=True)
		self.geo_frame2.pack_propagate(0)

		self.panel3 = Label(self.geo_frame2, image=self.image3, padx=5)
		self.panel3.pack(side='left', fill=BOTH, expand=True)

		#######################################################################################
		# Bottone chiusura
		#######################################################################################
		self.topButton = Button(self.top, text="Close", command = self.OnChildClose)
		self.topButton.pack(side='bottom')


	def OnChildClose(self):
		self.geo_frame.destroy()
		self.top.destroy()

	def OnCloseClick(self):
		self.destroy()



if __name__ == "__main__":
	logging.basicConfig(level=logging.ERROR)
	window = Window(None)
	window.title("WaveBEMapp - DTMB 5415")
	window.mainloop()
