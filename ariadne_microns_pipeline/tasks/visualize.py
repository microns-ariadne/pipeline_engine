import json
import luigi
import numpy as np
import os
import rh_logger
from scipy.ndimage import grey_erosion, grey_dilation
import sys

from ..parameters import VolumeParameter
from ..parameters import Volume, DatasetLocation
from .utilities import RequiresMixin, RunMixin
from ..targets.factory import TargetFactory
from .download_from_butterfly import DownloadFromButterflyTask

class VisualizeTaskMixin:
    
    volume = VolumeParameter(
        description="The volume to visualize")
    image_location=DatasetLocationParameter(
        description="The location of the image dataset")
    prob_location=DatasetLocationParameter(
        description="The location of the membrane probabilities")
    seeds_location=DatasetLocationParameter(
        description="The location of the watershed seed segmentation")
    seg_location=DatasetLocationParameter(
        description="The location of the segmentation")
    output_location=luigi.Parameter(
        description="The location for the .mp4 movie output")
    want_ply = luigi.BoolParameter(
        description="Visualize as a directory of polygons")
    
    def input(self):
        yield TargetFactory().get_volume_target(self.image_location,
                                                self.volume)
        yield TargetFactory().get_volume_target(self.prob_location,
                                                self.volume)
        yield TargetFactory().get_volume_target(self.seeds_location,
                                                self.volume)
        yield TargetFactory().get_volume_target(self.seg_location,
                                                self.volume)
    
    def output(self):
        if self.want_ply:
            return luigi.LocalTarget(self.output_location+".done")
        else:
            return luigi.LocalTarget(self.output_location)


class VisualizeRunMixin:
    #
    # Optional parameters
    #
    size_x_inches=luigi.FloatParameter(
        default=6,
        description="The target size of the output in the X direction "
        "in inches")
    size_y_inches=luigi.FloatParameter(
        default=6,
        description="The target size of the output in the Y direction "
        "in inches")
    first_frame=luigi.IntParameter(
        default=0,
        description="Start at this frame instead of frame # 0")
    end_frame=luigi.IntParameter(
        default=0,
        description="Stop the movie at this frame. Default=do all")
    xy_nm = luigi.FloatParameter(
        default=4,
        description="Voxel dimension in the x/y direction")
    z_nm = luigi.FloatParameter(
        default=30,
        description="Voxel dimension in the z direction")
    x_off = luigi.IntParameter(
        default=0,
        description="# of voxels in x direction to offset polygon")
    y_off = luigi.IntParameter(
        default=0,
        description="# of voxels in y direction to offset polygon")
    z_off = luigi.IntParameter(
        default=0,
        description="# of z-slices to offset polygon")
    
    def ariadne_run(self):
        if self.want_ply:
            self.render_seg()
        else:
            self.render_movie()
    
    def render_movie(self):
        '''Visualize the volume as a movie'''
        from matplotlib import animation
        from matplotlib.cm import ScalarMappable
        import matplotlib.pyplot as plt
        
        #
        # Some of this is from a recipe 
        # http://jakevdp.github.io/blog/2013/05/12/
        #        embedding-matplotlib-animations/
        #
        image_target, prob_target, seeds_target, seg_target = list(self.input())
        image_volume = image_target.imread()
        prob_volume = prob_target.imread()
        seeds_volume = seeds_target.imread()
        perm = np.random.RandomState(1234).permutation(np.max(seeds_volume)+1)
        seg_volume = seg_target.imread()
        figure = plt.figure(figsize=(self.size_x_inches, self.size_y_inches))
        ax = figure.add_subplot(1, 1, 1)
        sm = ScalarMappable(cmap='jet')
        sm.set_array(np.unique(seeds_volume))
        sm.autoscale()
        prob_sm = ScalarMappable(cmap='jet')
        prob_sm.set_array(np.linspace(0, 255, 256))
        prob_sm.autoscale()
        offset = self.first_frame
        end = self.end_frame if self.end_frame != 0 else image_volume.shape[0]
        
        def init_fig():
            ax.clear()
            ax.set_ylim(self.volume.height, 0)
            ax.set_xlim(0, self.volume.width)
        
        def animate(i1):
            i = i1 - offset
            rh_logger.logger.report_event("Processing frame %d" % i)
            ax.imshow(image_volume[i], cmap='gray')
            sv = sm.to_rgba(perm[seeds_volume[i]])
            sv[seeds_volume[i]==0, 3] = 0
            ax.imshow(sv)
            sv = sm.to_rgba(perm[seg_volume[i]])
            sv[seg_volume[i] == 0, 3] = 0
            sv[:, :, 3] *= .4
            ax.imshow(sv)
            prob = prob_sm.to_rgba(prob_volume[i])
            prob[:, :, 3] *= .4
            prob[prob_volume[i] < 128, 3] = 0
            ax.imshow(prob)
        
        anim = animation.FuncAnimation(figure, animate, init_func=init_fig,
                                       frames = end-offset,
                                       interval=1)
        plt.close(figure)
        anim.save(self.output().path, fps=1, extra_args=['-vcodec', 'libx264'])
    
    def render_seg(self):
        '''Render a segmentation as a set of .ply files'''
        image_target, prob_target, seeds_target, seg_target = list(self.input())
        seg_volume = seg_target.imread()
        #
        # Erode the edges of touching objects to get a mask of interiors
        #
        eroded = grey_erosion(seg_volume, 3)
        dilated = grey_dilation(seg_volume, 3)
        mask = (eroded == dilated) & (seg_volume != 0)
        #
        # The cube indices
        #
        #
        #       4        5
        #       ----------
        #      /|       /|
        #     / |      / |
        #    / (Z)    /  |
        #  7+--------+6  |
        #   |   |    |   |
        #   |  0+-(X)|---+1
        #   |  /     |  /
        #   |(Y)     | /
        #   |/       |/
        #   3--------2
        #
        # The edge indices
        #
        #
        #       ----4-----
        #      /|       /|
        #     7 |      5 |
        #    /  8     /  9
        #   +----6---+   |
        #   |   |    |   |
        #   |   +--0-|---+ 
        #   11 /     10 /
        #   | 3      | 1
        #   |/       |/
        #   +---2----+
        cube_offsets = np.array([[0, 0, 0],
                                 [0, 0, 1],
                                 [0, 1, 1],
                                 [0, 1, 0],
                                 [1, 0, 0],
                                 [1, 0, 1],
                                 [1, 1, 1],
                                 [1, 1, 0]])
        #
        # a 12 x 2 x 3 array giving the offsets of the vertices
        # that form the edge
        #
        edges = np.array([[[0, 0, 0], [0, 0, 1]], # 0
                          [[0, 0, 1], [0, 1, 1]], # 1
                          [[0, 1, 1], [0, 1, 0]], # 2
                          [[0, 1, 0], [0, 0, 0]], # 3
                          [[1, 0, 0], [1, 0, 1]], # 4
                          [[1, 0, 1], [1, 1, 1]], # 5
                          [[1, 1, 1], [1, 1, 0]], # 6
                          [[1, 1, 0], [1, 0, 0]], # 7
                          [[1, 0, 0], [0, 0, 0]], # 8
                          [[1, 0, 1], [0, 0, 1]], # 9
                          [[1, 1, 1], [0, 1, 1]], # 10
                          [[1, 1, 0], [0, 1, 0]]]) # 11
        
        edge_offsets = (edges[:, 0, :] + edges[:, 1, :]).astype(np.float32) / 2
        cubeindex = np.zeros(np.array(mask.shape)-1, np.uint8)
        cubeindex[mask[:-1, :-1, :-1]] += 1
        cubeindex[mask[:-1, :-1, 1:]] += 2
        cubeindex[mask[:-1, 1:, 1:]] += 4
        cubeindex[mask[:-1, 1:, :-1]] += 8
        cubeindex[mask[1:, :-1, :-1]] += 16
        cubeindex[mask[1:, :-1, 1:]] += 32
        cubeindex[mask[1:, 1:, 1:]] += 64
        cubeindex[mask[1:, 1:, :-1]] += 128
        
        z, y, x = [_.flatten() for _ in np.mgrid[0:cubeindex.shape[0], 
                                                 0:cubeindex.shape[1],
                                                 0:cubeindex.shape[2]]]
        cubeindex = cubeindex.flatten()
        #
        # Get rid of elements entirely in or out
        #
        cubeindex, z, y, x = \
            [_[n_triangles[cubeindex] > 0] for _ in cubeindex, z, y, x]
        coords = np.column_stack((z, y, x))
        #
        # Make the target array of coordinates
        #
        coord_idxs = np.hstack([[0], np.cumsum(n_triangles[cubeindex])])
        #
        # My old trick, two arrays, one giving the index into cubeindex
        # and a second into tri_table. Put them together and they
        # let you look up what you should put in the final vertex array
        #
        c_idx = np.zeros(coord_idxs[-1], int)
        c_idx[coord_idxs[1:-1]] = 1
        c_idx = np.cumsum(c_idx)
        t_idx = np.arange(coord_idxs[-1]) - coord_idxs[c_idx]
        #
        # And now all that's left to do is the lookup
        #
        triangles = \
            coords[c_idx, np.newaxis, :] +\
            edge_offsets[tri_table[cubeindex[c_idx], t_idx]]
        #
        # Get the label for each triangle. Either one or the other vertex
        # of the edge will be inside an object, but not both.
        #
        eoff = edges[tri_table[cubeindex[c_idx], t_idx, 0]]
        inside_1 = mask[z[c_idx] + eoff[:, 1, 0], 
                        y[c_idx] + eoff[:, 1, 1], 
                        x[c_idx] + eoff[:, 1, 2]].astype(np.uint8)
        idx = np.arange(len(inside_1))
        labels = seg_volume[z[c_idx] + eoff[idx, inside_1, 0],
                            y[c_idx] + eoff[idx, inside_1, 1],
                            x[c_idx] + eoff[idx, inside_1, 2]]
        order = np.argsort(labels)
        labels = labels[order]
        triangles = triangles[order]
        first_labels = \
            np.where(np.hstack([[True], labels[1:] != labels[:-1], [True]]))[0]
        #
        # Loop over each label
        #
        for tfirst, tlast in zip(first_labels[:-1], first_labels[1:]):
            #
            # To get into PLY format, find unique vertices and the correspondence
            # to each in the triangle list.
            #
            # 1) reshape the triangle list to be flattened from n x 3 x 3 to
            #    (n x 3) x 3
            # 2) use lexsort to get the order.
            # 3) find the first unique and do a trick to number them
            # 4) use the order to splatter them into the original order
            # 5) reshape the array as n x 3 x 3
            #
            l = labels[tfirst]
            t = triangles[tfirst:tlast].reshape((tlast - tfirst) * 3, 3)
            t[:, 2] += self.x_off
            t[:, 1] += self.y_off
            t[:, 1:] *= self.xy_nm / 1000.
            t[:, 0] += self.z_off
            t[:, 0] *= self.z_nm / 1000.
            order = np.lexsort([t[:, 2],  t[:, 1], t[:, 0]])
            t = t[order]
            first = np.hstack([[True], np.any(t[:-1] != t[1:], 1)])
            vertices = t[first]
            idx = np.cumsum(first)-1
            t = np.zeros(idx.shape[0], int)
            t[order] = idx
            t = t.reshape(tlast-tfirst, 3)
            #
            # Time to write it all out
            #
            filename = "label-%09d.ply" % l
            if not os.path.isdir(self.output_location):
                os.makedirs(self.output_location)
            with open(os.path.join(self.output_location, filename), "w") as fd:
                fd.write("ply\n")
                fd.write("format binary_%s_endian 1.0\n" % sys.byteorder)
                fd.write("comment Produced by the Ariadne-Microns pipeline\n")
                fd.write("element vertex %d\n" % len(vertices))
                fd.write("property float x\n")
                fd.write("property float y\n")
                fd.write("property float z\n")
                fd.write("element face %d\n" % len(t))
                fd.write("property list uchar uint vertex_index\n")
                fd.write("end_header\n")
                fd.write(vertices.astype(np.float32).data)
                #
                # Rework the triangles. There is a one-byte length of polygon
                # which is always "3". Then there are 3 4-byte vertex indices
                #
                triangle_buffer = np.zeros((len(t), 1 + 4 * 3), np.uint8)
                triangle_buffer[:, 0] = 3
                triangle_buffer[:, 1:] = np.frombuffer(
                    t.astype(np.uint32).data, np.uint8)\
                    .reshape(len(t), 4 * 3)
                fd.write(triangle_buffer.data)
        with open(self.output().path, "w") as fd:
            fd.write("done")

class VisualizeTask(VisualizeTaskMixin,
                    VisualizeRunMixin,
                    RequiresMixin,
                    RunMixin,
                    luigi.Task):
    task_namespace="ariadne_microns_pipeline"

class PipelineVisualizeTask(luigi.Task):
    '''Visualize the results within a pipeline directory'''
    
    experiment=luigi.Parameter(
        description="The name of the butterfly experiment")
    sample=luigi.Parameter(
        description="The name of the butterfly sample")
    dataset=luigi.Parameter(
        default="sem",
        description="The name of the butterfly dataset")
    channel=luigi.Parameter(
        default="raw",
        description="The name of the butterfly image channel, e.g. \"raw\"")
    directory=luigi.Parameter(
        description='The directory produced by a pipeline, e.g. '
        '".../ac3/unknown/sem/raw/26/26/1"')
    output_location=luigi.Parameter(
        description="The location for the movie file.")
    url=luigi.Parameter(
        default="http://localhost:2001/api",
        description="The URL of the butterfly server")
    #
    # Optional parameters
    #
    use_neuroproof=luigi.BoolParameter(
        description="Use the neuroproof segmentation")
    size_x_inches=luigi.FloatParameter(
        default=6,
        description="The target size of the output in the X direction "
        "in inches")
    size_y_inches=luigi.FloatParameter(
        default=6,
        description="The target size of the output in the Y direction "
        "in inches")
    first_frame=luigi.IntParameter(
        default=0,
        description="Start at this frame instead of frame # 0")
    end_frame=luigi.IntParameter(
        default=0,
        description="Stop the movie at this frame. Default=do all")
    want_ply=luigi.BoolParameter(
        description="Output polygon models instead of movie")
    
    task_namespace="ariadne_microns_pipeline"
    
    def requires(self):
        membrane_done = filter(lambda _: _.endswith("membrane.done"),
                               os.listdir(self.directory))
        assert len(membrane_done) == 1, \
               "Malformed directory: no membrane.done file"
        md_file = os.path.join(self.directory, membrane_done[0])
        md = json.load(open(md_file, "r"))
        dimensions = md["dimensions"]
        volume = Volume(md["x"], md["y"], md["z"], 
                        dimensions[2], dimensions[1], dimensions[0])
    
        image_location = DatasetLocation(
            [self.directory], 
            "image", 
            "{x:09d}_{y:09d}_{z:09d}_image")
        dfb_task = DownloadFromButterflyTask(
            experiment=self.experiment,
            sample=self.sample,
            dataset=self.dataset,
            channel=self.channel,
            volume=volume,
            destination=image_location,
            url=self.url)
        if self.use_neuroproof:
            seg_location = DatasetLocation(
                [self.directory], "neuroproof", 
                "{x:09d}_{y:09d}_{z:09d}_neuroproof")
        else:
            seg_location=DatasetLocation(
                    [self.directory], "segmentation", 
                    "{x:09d}_{y:09d}_{z:09d}_segmentation")
        vt = VisualizeTask(
            volume=volume,
            image_location=image_location,
            prob_location=DatasetLocation(
                [self.directory], "membrane", 
                "{x:09d}_{y:09d}_{z:09d}_membrane-membrane"),
            seeds_location=DatasetLocation(
                [self.directory], "seeds", "{x:09d}_{y:09d}_{z:09d}_seeds"),
            seg_location=seg_location,
            output_location=self.output_location,
            size_x_inches=self.size_x_inches,
            size_y_inches=self.size_y_inches,
            first_frame=self.first_frame,
            end_frame=self.end_frame,
            want_ply=self.want_ply)
        vt.set_requirement(dfb_task)
        yield vt

#
# Marching cubes tables taken from http://paulbourke.net/geometry/polygonise/
#
edge_table= np.array([
    0x0  , 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c,
    0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00,
    0x190, 0x99 , 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c,
    0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90,
    0x230, 0x339, 0x33 , 0x13a, 0x636, 0x73f, 0x435, 0x53c,
    0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30,
    0x3a0, 0x2a9, 0x1a3, 0xaa , 0x7a6, 0x6af, 0x5a5, 0x4ac,
    0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0,
    0x460, 0x569, 0x663, 0x76a, 0x66 , 0x16f, 0x265, 0x36c,
    0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60,
    0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0xff , 0x3f5, 0x2fc,
    0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0,
    0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x55 , 0x15c,
    0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950,
    0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0xcc ,
    0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0,
    0x8c0, 0x9c9, 0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc,
    0xcc , 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0,
    0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c,
    0x15c, 0x55 , 0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650,
    0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc,
    0x2fc, 0x3f5, 0xff , 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0,
    0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c,
    0x36c, 0x265, 0x16f, 0x66 , 0x76a, 0x663, 0x569, 0x460,
    0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac,
    0x4ac, 0x5a5, 0x6af, 0x7a6, 0xaa , 0x1a3, 0x2a9, 0x3a0,
    0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c,
    0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x33 , 0x339, 0x230,
    0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c,
    0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x99 , 0x190,
    0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c,
    0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x0])

tri_table = np.array(
[[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
 [0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
 [0, 1, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
 [1, 8, 3, 9, 8, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
 [1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
 [0, 8, 3, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
 [9, 2, 10, 0, 2, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
 [2, 8, 3, 2, 10, 8, 10, 9, 8, -1, -1, -1, -1, -1, -1, -1],
 [3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
 [0, 11, 2, 8, 11, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
 [1, 9, 0, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
 [1, 11, 2, 1, 9, 11, 9, 8, 11, -1, -1, -1, -1, -1, -1, -1],
 [3, 10, 1, 11, 10, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
 [0, 10, 1, 0, 8, 10, 8, 11, 10, -1, -1, -1, -1, -1, -1, -1],
 [3, 9, 0, 3, 11, 9, 11, 10, 9, -1, -1, -1, -1, -1, -1, -1],
 [9, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
 [4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
 [4, 3, 0, 7, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
 [0, 1, 9, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
 [4, 1, 9, 4, 7, 1, 7, 3, 1, -1, -1, -1, -1, -1, -1, -1],
 [1, 2, 10, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
 [3, 4, 7, 3, 0, 4, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1],
 [9, 2, 10, 9, 0, 2, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1],
 [2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4, -1, -1, -1, -1],
 [8, 4, 7, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
 [11, 4, 7, 11, 2, 4, 2, 0, 4, -1, -1, -1, -1, -1, -1, -1],
 [9, 0, 1, 8, 4, 7, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1],
 [4, 7, 11, 9, 4, 11, 9, 11, 2, 9, 2, 1, -1, -1, -1, -1],
 [3, 10, 1, 3, 11, 10, 7, 8, 4, -1, -1, -1, -1, -1, -1, -1],
 [1, 11, 10, 1, 4, 11, 1, 0, 4, 7, 11, 4, -1, -1, -1, -1],
 [4, 7, 8, 9, 0, 11, 9, 11, 10, 11, 0, 3, -1, -1, -1, -1],
 [4, 7, 11, 4, 11, 9, 9, 11, 10, -1, -1, -1, -1, -1, -1, -1],
 [9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
 [9, 5, 4, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
 [0, 5, 4, 1, 5, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
 [8, 5, 4, 8, 3, 5, 3, 1, 5, -1, -1, -1, -1, -1, -1, -1],
 [1, 2, 10, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
 [3, 0, 8, 1, 2, 10, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1],
 [5, 2, 10, 5, 4, 2, 4, 0, 2, -1, -1, -1, -1, -1, -1, -1],
 [2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8, -1, -1, -1, -1],
 [9, 5, 4, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
 [0, 11, 2, 0, 8, 11, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1],
 [0, 5, 4, 0, 1, 5, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1],
 [2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5, -1, -1, -1, -1],
 [10, 3, 11, 10, 1, 3, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1],
 [4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10, -1, -1, -1, -1],
 [5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3, -1, -1, -1, -1],
 [5, 4, 8, 5, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1],
 [9, 7, 8, 5, 7, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
 [9, 3, 0, 9, 5, 3, 5, 7, 3, -1, -1, -1, -1, -1, -1, -1],
 [0, 7, 8, 0, 1, 7, 1, 5, 7, -1, -1, -1, -1, -1, -1, -1],
 [1, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
 [9, 7, 8, 9, 5, 7, 10, 1, 2, -1, -1, -1, -1, -1, -1, -1],
 [10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3, -1, -1, -1, -1],
 [8, 0, 2, 8, 2, 5, 8, 5, 7, 10, 5, 2, -1, -1, -1, -1],
 [2, 10, 5, 2, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1],
 [7, 9, 5, 7, 8, 9, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1],
 [9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7, 11, -1, -1, -1, -1],
 [2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7, -1, -1, -1, -1],
 [11, 2, 1, 11, 1, 7, 7, 1, 5, -1, -1, -1, -1, -1, -1, -1],
 [9, 5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11, -1, -1, -1, -1],
 [5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0, -1],
 [11, 10, 0, 11, 0, 3, 10, 5, 0, 8, 0, 7, 5, 7, 0, -1],
 [11, 10, 5, 7, 11, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
 [10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
 [0, 8, 3, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
 [9, 0, 1, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
 [1, 8, 3, 1, 9, 8, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1],
 [1, 6, 5, 2, 6, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
 [1, 6, 5, 1, 2, 6, 3, 0, 8, -1, -1, -1, -1, -1, -1, -1],
 [9, 6, 5, 9, 0, 6, 0, 2, 6, -1, -1, -1, -1, -1, -1, -1],
 [5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8, -1, -1, -1, -1],
 [2, 3, 11, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
 [11, 0, 8, 11, 2, 0, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1],
 [0, 1, 9, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1],
 [5, 10, 6, 1, 9, 2, 9, 11, 2, 9, 8, 11, -1, -1, -1, -1],
 [6, 3, 11, 6, 5, 3, 5, 1, 3, -1, -1, -1, -1, -1, -1, -1],
 [0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6, -1, -1, -1, -1],
 [3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9, -1, -1, -1, -1],
 [6, 5, 9, 6, 9, 11, 11, 9, 8, -1, -1, -1, -1, -1, -1, -1],
 [5, 10, 6, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
 [4, 3, 0, 4, 7, 3, 6, 5, 10, -1, -1, -1, -1, -1, -1, -1],
 [1, 9, 0, 5, 10, 6, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1],
 [10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4, -1, -1, -1, -1],
 [6, 1, 2, 6, 5, 1, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1],
 [1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7, -1, -1, -1, -1],
 [8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6, -1, -1, -1, -1],
 [7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9, -1],
 [3, 11, 2, 7, 8, 4, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1],
 [5, 10, 6, 4, 7, 2, 4, 2, 0, 2, 7, 11, -1, -1, -1, -1],
 [0, 1, 9, 4, 7, 8, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1],
 [9, 2, 1, 9, 11, 2, 9, 4, 11, 7, 11, 4, 5, 10, 6, -1],
 [8, 4, 7, 3, 11, 5, 3, 5, 1, 5, 11, 6, -1, -1, -1, -1],
 [5, 1, 11, 5, 11, 6, 1, 0, 11, 7, 11, 4, 0, 4, 11, -1],
 [0, 5, 9, 0, 6, 5, 0, 3, 6, 11, 6, 3, 8, 4, 7, -1],
 [6, 5, 9, 6, 9, 11, 4, 7, 9, 7, 11, 9, -1, -1, -1, -1],
 [10, 4, 9, 6, 4, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
 [4, 10, 6, 4, 9, 10, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1],
 [10, 0, 1, 10, 6, 0, 6, 4, 0, -1, -1, -1, -1, -1, -1, -1],
 [8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 10, -1, -1, -1, -1],
 [1, 4, 9, 1, 2, 4, 2, 6, 4, -1, -1, -1, -1, -1, -1, -1],
 [3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4, -1, -1, -1, -1],
 [0, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
 [8, 3, 2, 8, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1],
 [10, 4, 9, 10, 6, 4, 11, 2, 3, -1, -1, -1, -1, -1, -1, -1],
 [0, 8, 2, 2, 8, 11, 4, 9, 10, 4, 10, 6, -1, -1, -1, -1],
 [3, 11, 2, 0, 1, 6, 0, 6, 4, 6, 1, 10, -1, -1, -1, -1],
 [6, 4, 1, 6, 1, 10, 4, 8, 1, 2, 1, 11, 8, 11, 1, -1],
 [9, 6, 4, 9, 3, 6, 9, 1, 3, 11, 6, 3, -1, -1, -1, -1],
 [8, 11, 1, 8, 1, 0, 11, 6, 1, 9, 1, 4, 6, 4, 1, -1],
 [3, 11, 6, 3, 6, 0, 0, 6, 4, -1, -1, -1, -1, -1, -1, -1],
 [6, 4, 8, 11, 6, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
 [7, 10, 6, 7, 8, 10, 8, 9, 10, -1, -1, -1, -1, -1, -1, -1],
 [0, 7, 3, 0, 10, 7, 0, 9, 10, 6, 7, 10, -1, -1, -1, -1],
 [10, 6, 7, 1, 10, 7, 1, 7, 8, 1, 8, 0, -1, -1, -1, -1],
 [10, 6, 7, 10, 7, 1, 1, 7, 3, -1, -1, -1, -1, -1, -1, -1],
 [1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7, -1, -1, -1, -1],
 [2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9, -1],
 [7, 8, 0, 7, 0, 6, 6, 0, 2, -1, -1, -1, -1, -1, -1, -1],
 [7, 3, 2, 6, 7, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
 [2, 3, 11, 10, 6, 8, 10, 8, 9, 8, 6, 7, -1, -1, -1, -1],
 [2, 0, 7, 2, 7, 11, 0, 9, 7, 6, 7, 10, 9, 10, 7, -1],
 [1, 8, 0, 1, 7, 8, 1, 10, 7, 6, 7, 10, 2, 3, 11, -1],
 [11, 2, 1, 11, 1, 7, 10, 6, 1, 6, 7, 1, -1, -1, -1, -1],
 [8, 9, 6, 8, 6, 7, 9, 1, 6, 11, 6, 3, 1, 3, 6, -1],
 [0, 9, 1, 11, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
 [7, 8, 0, 7, 0, 6, 3, 11, 0, 11, 6, 0, -1, -1, -1, -1],
 [7, 11, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
 [7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
 [3, 0, 8, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
 [0, 1, 9, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
 [8, 1, 9, 8, 3, 1, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1],
 [10, 1, 2, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
 [1, 2, 10, 3, 0, 8, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1],
 [2, 9, 0, 2, 10, 9, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1],
 [6, 11, 7, 2, 10, 3, 10, 8, 3, 10, 9, 8, -1, -1, -1, -1],
 [7, 2, 3, 6, 2, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
 [7, 0, 8, 7, 6, 0, 6, 2, 0, -1, -1, -1, -1, -1, -1, -1],
 [2, 7, 6, 2, 3, 7, 0, 1, 9, -1, -1, -1, -1, -1, -1, -1],
 [1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6, -1, -1, -1, -1],
 [10, 7, 6, 10, 1, 7, 1, 3, 7, -1, -1, -1, -1, -1, -1, -1],
 [10, 7, 6, 1, 7, 10, 1, 8, 7, 1, 0, 8, -1, -1, -1, -1],
 [0, 3, 7, 0, 7, 10, 0, 10, 9, 6, 10, 7, -1, -1, -1, -1],
 [7, 6, 10, 7, 10, 8, 8, 10, 9, -1, -1, -1, -1, -1, -1, -1],
 [6, 8, 4, 11, 8, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
 [3, 6, 11, 3, 0, 6, 0, 4, 6, -1, -1, -1, -1, -1, -1, -1],
 [8, 6, 11, 8, 4, 6, 9, 0, 1, -1, -1, -1, -1, -1, -1, -1],
 [9, 4, 6, 9, 6, 3, 9, 3, 1, 11, 3, 6, -1, -1, -1, -1],
 [6, 8, 4, 6, 11, 8, 2, 10, 1, -1, -1, -1, -1, -1, -1, -1],
 [1, 2, 10, 3, 0, 11, 0, 6, 11, 0, 4, 6, -1, -1, -1, -1],
 [4, 11, 8, 4, 6, 11, 0, 2, 9, 2, 10, 9, -1, -1, -1, -1],
 [10, 9, 3, 10, 3, 2, 9, 4, 3, 11, 3, 6, 4, 6, 3, -1],
 [8, 2, 3, 8, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1],
 [0, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
 [1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8, -1, -1, -1, -1],
 [1, 9, 4, 1, 4, 2, 2, 4, 6, -1, -1, -1, -1, -1, -1, -1],
 [8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 10, 1, -1, -1, -1, -1],
 [10, 1, 0, 10, 0, 6, 6, 0, 4, -1, -1, -1, -1, -1, -1, -1],
 [4, 6, 3, 4, 3, 8, 6, 10, 3, 0, 3, 9, 10, 9, 3, -1],
 [10, 9, 4, 6, 10, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
 [4, 9, 5, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
 [0, 8, 3, 4, 9, 5, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1],
 [5, 0, 1, 5, 4, 0, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1],
 [11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5, -1, -1, -1, -1],
 [9, 5, 4, 10, 1, 2, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1],
 [6, 11, 7, 1, 2, 10, 0, 8, 3, 4, 9, 5, -1, -1, -1, -1],
 [7, 6, 11, 5, 4, 10, 4, 2, 10, 4, 0, 2, -1, -1, -1, -1],
 [3, 4, 8, 3, 5, 4, 3, 2, 5, 10, 5, 2, 11, 7, 6, -1],
 [7, 2, 3, 7, 6, 2, 5, 4, 9, -1, -1, -1, -1, -1, -1, -1],
 [9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7, -1, -1, -1, -1],
 [3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0, -1, -1, -1, -1],
 [6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8, -1],
 [9, 5, 4, 10, 1, 6, 1, 7, 6, 1, 3, 7, -1, -1, -1, -1],
 [1, 6, 10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4, -1],
 [4, 0, 10, 4, 10, 5, 0, 3, 10, 6, 10, 7, 3, 7, 10, -1],
 [7, 6, 10, 7, 10, 8, 5, 4, 10, 4, 8, 10, -1, -1, -1, -1],
 [6, 9, 5, 6, 11, 9, 11, 8, 9, -1, -1, -1, -1, -1, -1, -1],
 [3, 6, 11, 0, 6, 3, 0, 5, 6, 0, 9, 5, -1, -1, -1, -1],
 [0, 11, 8, 0, 5, 11, 0, 1, 5, 5, 6, 11, -1, -1, -1, -1],
 [6, 11, 3, 6, 3, 5, 5, 3, 1, -1, -1, -1, -1, -1, -1, -1],
 [1, 2, 10, 9, 5, 11, 9, 11, 8, 11, 5, 6, -1, -1, -1, -1],
 [0, 11, 3, 0, 6, 11, 0, 9, 6, 5, 6, 9, 1, 2, 10, -1],
 [11, 8, 5, 11, 5, 6, 8, 0, 5, 10, 5, 2, 0, 2, 5, -1],
 [6, 11, 3, 6, 3, 5, 2, 10, 3, 10, 5, 3, -1, -1, -1, -1],
 [5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2, -1, -1, -1, -1],
 [9, 5, 6, 9, 6, 0, 0, 6, 2, -1, -1, -1, -1, -1, -1, -1],
 [1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8, -1],
 [1, 5, 6, 2, 1, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
 [1, 3, 6, 1, 6, 10, 3, 8, 6, 5, 6, 9, 8, 9, 6, -1],
 [10, 1, 0, 10, 0, 6, 9, 5, 0, 5, 6, 0, -1, -1, -1, -1],
 [0, 3, 8, 5, 6, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
 [10, 5, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
 [11, 5, 10, 7, 5, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
 [11, 5, 10, 11, 7, 5, 8, 3, 0, -1, -1, -1, -1, -1, -1, -1],
 [5, 11, 7, 5, 10, 11, 1, 9, 0, -1, -1, -1, -1, -1, -1, -1],
 [10, 7, 5, 10, 11, 7, 9, 8, 1, 8, 3, 1, -1, -1, -1, -1],
 [11, 1, 2, 11, 7, 1, 7, 5, 1, -1, -1, -1, -1, -1, -1, -1],
 [0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2, 11, -1, -1, -1, -1],
 [9, 7, 5, 9, 2, 7, 9, 0, 2, 2, 11, 7, -1, -1, -1, -1],
 [7, 5, 2, 7, 2, 11, 5, 9, 2, 3, 2, 8, 9, 8, 2, -1],
 [2, 5, 10, 2, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1],
 [8, 2, 0, 8, 5, 2, 8, 7, 5, 10, 2, 5, -1, -1, -1, -1],
 [9, 0, 1, 5, 10, 3, 5, 3, 7, 3, 10, 2, -1, -1, -1, -1],
 [9, 8, 2, 9, 2, 1, 8, 7, 2, 10, 2, 5, 7, 5, 2, -1],
 [1, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
 [0, 8, 7, 0, 7, 1, 1, 7, 5, -1, -1, -1, -1, -1, -1, -1],
 [9, 0, 3, 9, 3, 5, 5, 3, 7, -1, -1, -1, -1, -1, -1, -1],
 [9, 8, 7, 5, 9, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
 [5, 8, 4, 5, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1],
 [5, 0, 4, 5, 11, 0, 5, 10, 11, 11, 3, 0, -1, -1, -1, -1],
 [0, 1, 9, 8, 4, 10, 8, 10, 11, 10, 4, 5, -1, -1, -1, -1],
 [10, 11, 4, 10, 4, 5, 11, 3, 4, 9, 4, 1, 3, 1, 4, -1],
 [2, 5, 1, 2, 8, 5, 2, 11, 8, 4, 5, 8, -1, -1, -1, -1],
 [0, 4, 11, 0, 11, 3, 4, 5, 11, 2, 11, 1, 5, 1, 11, -1],
 [0, 2, 5, 0, 5, 9, 2, 11, 5, 4, 5, 8, 11, 8, 5, -1],
 [9, 4, 5, 2, 11, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
 [2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4, -1, -1, -1, -1],
 [5, 10, 2, 5, 2, 4, 4, 2, 0, -1, -1, -1, -1, -1, -1, -1],
 [3, 10, 2, 3, 5, 10, 3, 8, 5, 4, 5, 8, 0, 1, 9, -1],
 [5, 10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2, -1, -1, -1, -1],
 [8, 4, 5, 8, 5, 3, 3, 5, 1, -1, -1, -1, -1, -1, -1, -1],
 [0, 4, 5, 1, 0, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
 [8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5, -1, -1, -1, -1],
 [9, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
 [4, 11, 7, 4, 9, 11, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1],
 [0, 8, 3, 4, 9, 7, 9, 11, 7, 9, 10, 11, -1, -1, -1, -1],
 [1, 10, 11, 1, 11, 4, 1, 4, 0, 7, 4, 11, -1, -1, -1, -1],
 [3, 1, 4, 3, 4, 8, 1, 10, 4, 7, 4, 11, 10, 11, 4, -1],
 [4, 11, 7, 9, 11, 4, 9, 2, 11, 9, 1, 2, -1, -1, -1, -1],
 [9, 7, 4, 9, 11, 7, 9, 1, 11, 2, 11, 1, 0, 8, 3, -1],
 [11, 7, 4, 11, 4, 2, 2, 4, 0, -1, -1, -1, -1, -1, -1, -1],
 [11, 7, 4, 11, 4, 2, 8, 3, 4, 3, 2, 4, -1, -1, -1, -1],
 [2, 9, 10, 2, 7, 9, 2, 3, 7, 7, 4, 9, -1, -1, -1, -1],
 [9, 10, 7, 9, 7, 4, 10, 2, 7, 8, 7, 0, 2, 0, 7, -1],
 [3, 7, 10, 3, 10, 2, 7, 4, 10, 1, 10, 0, 4, 0, 10, -1],
 [1, 10, 2, 8, 7, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
 [4, 9, 1, 4, 1, 7, 7, 1, 3, -1, -1, -1, -1, -1, -1, -1],
 [4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1, -1, -1, -1, -1],
 [4, 0, 3, 7, 4, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
 [4, 8, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
 [9, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
 [3, 0, 9, 3, 9, 11, 11, 9, 10, -1, -1, -1, -1, -1, -1, -1],
 [0, 1, 10, 0, 10, 8, 8, 10, 11, -1, -1, -1, -1, -1, -1, -1],
 [3, 1, 10, 11, 3, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
 [1, 2, 11, 1, 11, 9, 9, 11, 8, -1, -1, -1, -1, -1, -1, -1],
 [3, 0, 9, 3, 9, 11, 1, 2, 9, 2, 11, 9, -1, -1, -1, -1],
 [0, 2, 11, 8, 0, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
 [3, 2, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
 [2, 3, 8, 2, 8, 10, 10, 8, 9, -1, -1, -1, -1, -1, -1, -1],
 [9, 10, 2, 0, 9, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
 [2, 3, 8, 2, 8, 10, 0, 1, 8, 1, 10, 8, -1, -1, -1, -1],
 [1, 10, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
 [1, 3, 8, 9, 1, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
 [0, 9, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
 [0, 3, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
 [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]])
n_triangles = np.sum(tri_table != -1, 1) / 3
tri_table = tri_table[:, :-1].reshape(256, 5, 3)