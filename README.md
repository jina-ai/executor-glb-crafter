# GlbCrafter

**GlbCrafter** is an Executor that receives Documents with `.glb` files. Documents can either have a uri that 
represents the path to the `.glb` file or a blob obtained after converting the uri to buffer.
**GlbCrafter** samples points from the 3D mesh object to create a point cloud and puts them in the blob attribute of 
the Document.
