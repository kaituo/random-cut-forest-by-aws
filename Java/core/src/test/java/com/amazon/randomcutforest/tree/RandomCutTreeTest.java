/*
 * Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License").
 * You may not use this file except in compliance with the License.
 * A copy of the License is located at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * or in the "license" file accompanying this file. This file is distributed
 * on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
 * express or implied. See the License for the specific language governing
 * permissions and limitations under the License.
 */

package com.amazon.randomcutforest.tree;

import static com.amazon.randomcutforest.CommonUtils.toDoubleArray;
import static com.amazon.randomcutforest.CommonUtils.toFloatArray;
import static com.amazon.randomcutforest.tree.AbstractNodeStore.Null;
import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.closeTo;
import static org.hamcrest.Matchers.is;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import com.amazon.randomcutforest.config.Config;
import com.amazon.randomcutforest.sampler.Weighted;
import com.amazon.randomcutforest.store.PointStore;

public class RandomCutTreeTest {

    private static final double EPSILON = 1e-8;

    private Random rng;
    private RandomCutTree tree;
    private NodeStoreSmall nodeStoreSmall;

    @BeforeEach
    public void setUp() {
        rng = mock(Random.class);
        PointStore pointStoreFloat = new PointStore.Builder().indexCapacity(100).capacity(100).initialSize(100)
                .dimensions(2).build();
        tree = RandomCutTree.builder().random(rng).centerOfMassEnabled(true).pointStoreView(pointStoreFloat)
                .storeSequenceIndexesEnabled(true).storeParent(true).dimension(2).build();

        // Create the following tree structure (in the second diagram., backticks denote
        // cuts)
        // The leaf point 0,1 has mass 2, all other nodes have mass 1.
        //
        // /\
        // / \
        // -1,-1 / \
        // / \
        // /\ 1,1
        // / \
        // -1,0 0,1
        //
        //
        // 0,1 1,1
        // ----------*---------*
        // | ` | ` |
        // | ` | ` |
        // | ` | ` |
        // -1,0 *-------------------|
        // | |
        // |```````````````````|
        // | |
        // -1,-1 *--------------------
        //
        // We choose the insertion order and random draws carefully so that each split
        // divides its parent in half.
        // The random values are used to set the cut dimensions and values.

        assertEquals(pointStoreFloat.add(new float[] { -1, -1 }, 1), 0);
        assertEquals(pointStoreFloat.add(new float[] { 1, 1 }, 2), 1);
        assertEquals(pointStoreFloat.add(new float[] { -1, 0 }, 3), 2);
        assertEquals(pointStoreFloat.add(new float[] { 0, 1 }, 4), 3);
        assertEquals(pointStoreFloat.add(new float[] { 0, 1 }, 5), 4);
        assertEquals(pointStoreFloat.add(new float[] { 0, 0 }, 6), 5);

        assertThrows(IllegalArgumentException.class, () -> tree.deletePoint(0, 1));
        tree.addPoint(0, 1);
        tree.deletePoint(0, 1);
        assertTrue(tree.root == Null);

        tree.addPoint(0, 1);
        when(rng.nextDouble()).thenReturn(0.625);
        tree.addPoint(1, 2);

        when(rng.nextDouble()).thenReturn(0.5);
        tree.addPoint(2, 3);

        when(rng.nextDouble()).thenReturn(0.25);
        tree.addPoint(3, 4);

        // add mass to 0,1
        tree.addPoint(4, 5);
    }

    @Test
    public void testConfig() {
        assertThrows(IllegalArgumentException.class, () -> tree.setBoundingBoxCacheFraction(-0.5));
        assertThrows(IllegalArgumentException.class, () -> tree.setBoundingBoxCacheFraction(2.0));
        assertThrows(IllegalArgumentException.class, () -> tree.setConfig("foo", 0));
        assertThrows(IllegalArgumentException.class, () -> tree.getConfig("bar"));
        assertEquals(tree.getConfig(Config.BOUNDING_BOX_CACHE_FRACTION), 1.0);
        assertThrows(IllegalArgumentException.class, () -> tree.setConfig(Config.BOUNDING_BOX_CACHE_FRACTION, true));
        tree.setConfig(Config.BOUNDING_BOX_CACHE_FRACTION, 0.2);
    }

    @Test
    public void testConfigDelete() {
        PointStore pointStore = mock(PointStore.class);
        tree = RandomCutTree.builder().random(rng).centerOfMassEnabled(true).pointStoreView(pointStore)
                .storeSequenceIndexesEnabled(true).storeParent(true).dimension(3).build();

        when(pointStore.getNumericVector(any(Integer.class))).thenReturn(new float[] { 0 }).thenReturn(new float[3])
                .thenReturn(new float[3]);
        tree.addPoint(0, 1);
        // fails vor dimension
        assertThrows(IllegalArgumentException.class, () -> tree.deletePoint(0, 1));
        assertThrows(IllegalArgumentException.class, () -> tree.deletePoint(2, 1));
        // wrong sequence index
        assertThrows(IllegalArgumentException.class, () -> tree.deletePoint(0, 2));
        // state is corrupted
        assertThrows(IllegalArgumentException.class, () -> tree.deletePoint(0, 1));
    }

    @Test
    public void testConfigAdd() {
        PointStore pointStore = mock(PointStore.class);
        float[] test = new float[] { 1.119f, 0f, -3.11f, 100f };
        float[] copies = new float[] { 0, 17, 0, 0 };
        tree = RandomCutTree.builder().random(rng).centerOfMassEnabled(true).pointStoreView(pointStore)
                .centerOfMassEnabled(true).storeSequenceIndexesEnabled(true).storeParent(true).dimension(4).build();
        when(pointStore.getNumericVector(any(Integer.class))).thenReturn(new float[0]).thenReturn(test)
                .thenReturn(new float[0]).thenReturn(new float[4]).thenReturn(new float[4]).thenReturn(new float[5])
                .thenReturn(copies).thenReturn(test).thenReturn(copies).thenReturn(copies).thenReturn(test);

        // cannot have partial addition to empty tree
        assertThrows(IllegalArgumentException.class, () -> tree.addPointToPartialTree(0, 1));
        // the following does not consume any points
        tree.addPoint(0, 1);
        // consumes from pointstore but gets 0 length vector
        assertThrows(IllegalArgumentException.class, () -> tree.getPointSum(tree.getRoot()));
        // passes, consumes pointstore
        assertArrayEquals(tree.getPointSum(tree.getRoot()), test);
        // in the sequel point is [0,0,0,0] fails because old point appears to have 5
        // dimensions
        assertThrows(IllegalArgumentException.class, () -> tree.addPoint(1, 1));
        // this invocation succeeds, but points are same
        tree.addPoint(1, 1);
        assertTrue(tree.isLeaf(tree.getRoot()));
        // dimension = 5
        assertThrows(IllegalArgumentException.class, () -> tree.addPoint(2, 1));
        // switch the vector
        assertArrayEquals(tree.getPointSum(tree.getRoot()), new float[] { 0, 34, 0, 0 });
        // adding test, consumes the copy
        tree.addPoint(2, 1);
        assertEquals(tree.getMass(), 3);
        assertArrayEquals(tree.getPointSum(tree.getRoot()), new float[] { 1.119f, 34, -3.11f, 100 }, 1e-3f);
    }

    @Test
    public void testCut() {
        PointStore pointStore = mock(PointStore.class);
        Random random = mock(Random.class);
        tree = RandomCutTree.builder().random(rng).centerOfMassEnabled(true).pointStoreView(pointStore).random(random)
                .storeSequenceIndexesEnabled(true).storeParent(true).dimension(1).build();
        when(pointStore.getNumericVector(any(Integer.class))).thenReturn(new float[1]).thenReturn(new float[] { 1 })
                .thenReturn(new float[] { 0 }).thenReturn(new float[] { 0 }).thenReturn(new float[] { 2 })
                .thenReturn(new float[] { 1 }).thenReturn(new float[0]).thenReturn(new float[] { 2 })
                .thenReturn(new float[] { 1 }).thenReturn(new float[1]);
        // testing the cut assumptions -- the values should not be 1 or larger, but is
        // useful for testing
        when(random.nextDouble()).thenReturn(1.2).thenReturn(1.5).thenReturn(1.5).thenReturn(0.0);
        // following does not query pointstore
        tree.addPoint(0, 1);
        // following tries to add [0.0], and discovers point index 0 is [1.0]
        tree.addPoint(1, 1);
        assertTrue(tree.getCutValue(tree.getRoot()) == (double) Math.nextAfter(1.0f, 0.0));

        assertThrows(IllegalArgumentException.class, () -> tree.addPoint(1, 2)); // copy
        tree.addPoint(1, 2); // passes
        assertTrue(tree.getRoot() == 0);
        assertTrue(tree.getCutValue(0) == (double) Math.nextAfter(1.0f, 0.0));
        assertTrue(tree.getCutValue(1) == (double) Math.nextAfter(2.0f, 1.0));
        assertFalse(tree.checkStrictlyContains(1, new float[] { 2 }));
        assertTrue(tree.checkStrictlyContains(1, new float[] { 1.001f }));
    }

    /**
     * Verify that the tree has the form described in the setUp method.
     */
    @Test
    public void testInitialTreeState() {
        int node = tree.getRoot();
        // the second double[] is intentional
        IBoundingBoxView expectedBox = new BoundingBox(new float[] { -1, -1 }).getMergedBox(new float[] { 1, 1 });
        assertThat(tree.getBox(node), is(expectedBox));
        assertThat(tree.getCutDimension(node), is(1));
        assertThat(tree.getCutValue(node), closeTo(-0.5, EPSILON));
        assertThat(tree.getMass(), is(5));
        assertArrayEquals(new double[] { -1, 2 }, toDoubleArray(tree.getPointSum(node)), EPSILON);
        assertThat(tree.isLeaf(tree.getLeftChild(node)), is(true));
        assertThat(tree.pointStoreView.getNumericVector(tree.getPointIndex(tree.getLeftChild(node))),
                is(new float[] { -1, -1 }));
        assertThat(tree.getMass(tree.getLeftChild(node)), is(1));
        assertEquals(tree.getSequenceMap(tree.getPointIndex(tree.getLeftChild(node))).get(1L), 1);
        // testing inappropriate
        assertThrows(IllegalArgumentException.class, () -> tree.getLeftChild(Integer.MAX_VALUE));
        assertThrows(IllegalArgumentException.class, () -> tree.getRightChild(500));
        assertThrows(IllegalArgumentException.class, () -> tree.getCutValue(-1));
        assertThrows(IllegalArgumentException.class, () -> tree.getCutDimension(-1));
        // pointIndex should have a value at least as large as number of leaves
        assertThrows(IllegalArgumentException.class, () -> tree.getPointIndex(0));

        NodeStoreSmall nodeStoreSmall = (NodeStoreSmall) tree.nodeStore;
        assert (nodeStoreSmall.getParentIndex(tree.getRightChild(node)) == node);
        node = tree.getRightChild(node);
        expectedBox = new BoundingBox(new float[] { -1, 0 }).getMergedBox(new BoundingBox(new float[] { 1, 1 }));
        assertThat(tree.getBox(node), is(expectedBox));
        assertThat(tree.getCutDimension(node), is(0));
        assertThat(tree.getCutValue(node), closeTo(0.5, EPSILON));
        assertThat(tree.getMass(node), is(4));
        assertArrayEquals(new double[] { 0.0, 3.0 }, toDoubleArray(tree.getPointSum(node)), EPSILON);

        assertThat(tree.isLeaf(tree.getRightChild(node)), is(true));
        assertThat(tree.pointStoreView.getNumericVector(tree.getPointIndex(tree.getRightChild(node))),
                is(new float[] { 1, 1 }));
        assertThat(tree.getMass(tree.getRightChild(node)), is(1));
        assertEquals(tree.getSequenceMap(tree.getPointIndex(tree.getRightChild(node))).get(2L), 1);

        assert (nodeStoreSmall.getParentIndex(tree.getLeftChild(node)) == node);
        node = tree.getLeftChild(node);
        expectedBox = new BoundingBox(new float[] { -1, 0 }).getMergedBox(new float[] { 0, 1 });
        assertThat(tree.getBox(node), is(expectedBox));

        assertThat(tree.getCutDimension(node), is(0));
        assertThat(tree.getCutValue(node), closeTo(-0.5, EPSILON));
        assertThat(tree.getMass(node), is(3));
        assertArrayEquals(new double[] { -1.0, 2.0 }, toDoubleArray(tree.getPointSum(node)), EPSILON);
        assertThat(tree.isLeaf(tree.getLeftChild(node)), is(true));
        assertThat(tree.pointStoreView.getNumericVector(tree.getPointIndex(tree.getLeftChild(node))),
                is(new float[] { -1, 0 }));
        assertThat(tree.getMass(tree.getLeftChild(node)), is(1));
        assertEquals(tree.getSequenceMap(tree.getPointIndex(tree.getLeftChild(node))).get(3L), 1);

        assertThat(tree.isLeaf(tree.getRightChild(node)), is(true));
        assertThat(tree.pointStoreView.getNumericVector(tree.getPointIndex(tree.getRightChild(node))),
                is(new float[] { 0, 1 }));
        assertThat(tree.getMass(tree.getRightChild(node)), is(2));
        assertEquals(tree.getSequenceMap(tree.getPointIndex(tree.getRightChild(node))).get(4L), 1);
        assertEquals(tree.getSequenceMap(tree.getPointIndex(tree.getRightChild(node))).get(5L), 1);
        assertThrows(IllegalArgumentException.class, () -> tree.deletePoint(5, 6));
    }

    @Test
    public void testDeletePointWithLeafSibling() {
        tree.deletePoint(2, 3);

        // root node bounding box and cut remains unchanged, mass and centerOfMass are
        // updated

        int node = tree.getRoot();
        IBoundingBoxView expectedBox = new BoundingBox(new float[] { -1, -1 }).getMergedBox(new float[] { 1, 1 });
        assertThat(tree.getBox(node), is(expectedBox));
        assertThat(tree.getCutDimension(node), is(1));
        assertThat(tree.getCutValue(node), closeTo(-0.5, EPSILON));
        assertThat(tree.getMass(), is(4));

        assertArrayEquals(new double[] { 0.0, 2.0 }, toDoubleArray(tree.getPointSum(node)), EPSILON);

        assertThat(tree.isLeaf(tree.getLeftChild(node)), is(true));
        assertThat(tree.pointStoreView.getNumericVector(tree.getPointIndex(tree.getLeftChild(node))),
                is(new float[] { -1, -1 }));
        assertThat(tree.getMass(tree.getLeftChild(node)), is(1));
        assertEquals(tree.getSequenceMap(tree.getPointIndex(tree.getLeftChild(node))).get(1L), 1);
        // sibling node moves up and bounding box recomputed

        NodeStoreSmall nodeStoreSmall = (NodeStoreSmall) tree.nodeStore;
        assert (nodeStoreSmall.getParentIndex(tree.getRightChild(node)) == node);
        node = tree.getRightChild(node);
        expectedBox = new BoundingBox(new float[] { 0, 1 }).getMergedBox(new float[] { 1, 1 });
        assertThat(tree.getBox(node), is(expectedBox));
        assertThat(tree.getCutDimension(node), is(0));
        assertThat(tree.getCutValue(node), closeTo(0.5, EPSILON));
        assertThat(tree.getMass(node), is(3));
        assertArrayEquals(new double[] { 1.0, 3.0 }, toDoubleArray(tree.getPointSum(node)), EPSILON);

        assertThat(tree.isLeaf(tree.getLeftChild(node)), is(true));
        assertThat(tree.pointStoreView.getNumericVector(tree.getPointIndex(tree.getLeftChild(node))),
                is(new float[] { 0, 1 }));
        assertThat(tree.getMass(tree.getLeftChild(node)), is(2));
        assertEquals(tree.getSequenceMap(tree.getPointIndex(tree.getLeftChild(node))).get(4L), 1);
        assertEquals(tree.getSequenceMap(tree.getPointIndex(tree.getLeftChild(node))).get(5L), 1);

        assertThat(tree.isLeaf(tree.getRightChild(node)), is(true));
        assertThat(tree.pointStoreView.getNumericVector(tree.getPointIndex(tree.getRightChild(node))),
                is(new float[] { 1, 1 }));
        assertThat(tree.getMass(tree.getRightChild(node)), is(1));
        assertEquals(tree.getSequenceMap(tree.getPointIndex(tree.getRightChild(node))).get(2L), 1);
    }

    @Test
    public void testDeletePointWithNonLeafSibling() {
        tree.deletePoint(1, 2);

        // root node bounding box recomputed

        int node = tree.getRoot();
        IBoundingBoxView expectedBox = new BoundingBox(new float[] { -1, -1 }).getMergedBox(new float[] { 0, 1 });
        assertThat(tree.getBox(node), is(expectedBox));
        assertThat(tree.getCutDimension(node), is(1));
        assertThat(tree.getCutValue(node), closeTo(-0.5, EPSILON));
        assertThat(tree.getMass(), is(4));

        assertThat(tree.isLeaf(tree.getLeftChild(node)), is(true));
        assertThat(tree.pointStoreView.getNumericVector(tree.getPointIndex(tree.getLeftChild(node))),
                is(new float[] { -1, -1 }));
        assertThat(tree.getMass(tree.getLeftChild(node)), is(1));
        assertEquals(tree.getSequenceMap(tree.getPointIndex(tree.getLeftChild(node))).get(1L), 1);

        // sibling node moves up and bounding box stays the same
        NodeStoreSmall nodeStoreSmall = (NodeStoreSmall) tree.nodeStore;
        assert (nodeStoreSmall.getParentIndex(tree.getRightChild(node)) == node);
        node = tree.getRightChild(node);
        expectedBox = new BoundingBox(new float[] { -1, 0 }).getMergedBox(new float[] { 0, 1 });
        assertThat(tree.getBox(node), is(expectedBox));
        assertThat(tree.getCutDimension(node), is(0));
        assertThat(tree.getCutValue(node), closeTo(-0.5, EPSILON));

        assertThat(tree.isLeaf(tree.getLeftChild(node)), is(true));
        assertThat(tree.pointStoreView.getNumericVector(tree.getPointIndex(tree.getLeftChild(node))),
                is(new float[] { -1, 0 }));
        assertThat(tree.getMass(tree.getLeftChild(node)), is(1));
        assertEquals(tree.getSequenceMap(tree.getPointIndex(tree.getLeftChild(node))).get(3L), 1);

        assertThat(tree.isLeaf(tree.getRightChild(node)), is(true));
        assertThat(tree.pointStoreView.getNumericVector(tree.getPointIndex(tree.getRightChild(node))),
                is(new float[] { 0, 1 }));
        assertThat(tree.getMass(tree.getRightChild(node)), is(2));
        assertEquals(tree.getSequenceMap(tree.getPointIndex(tree.getRightChild(node))).get(4L), 1);
        assertEquals(tree.getSequenceMap(tree.getPointIndex(tree.getRightChild(node))).get(5L), 1);
    }

    @Test
    public void testDeletePointWithMassGreaterThan1() {
        tree.deletePoint(3, 4);

        // same as initial state except mass at 0,1 is 1

        int node = tree.getRoot();
        IBoundingBoxView expectedBox = new BoundingBox(new float[] { -1, -1 }).getMergedBox(new float[] { 1, 1 });
        assertThat(tree.getBox(node), is(expectedBox));
        assertThat(tree.getCutDimension(node), is(1));
        assertThat(tree.getCutValue(node), closeTo(-0.5, EPSILON));
        assertThat(tree.getMass(), is(4));

        assertThat(tree.isLeaf(tree.getLeftChild(node)), is(true));
        assertThat(tree.pointStoreView.getNumericVector(tree.getPointIndex(tree.getLeftChild(node))),
                is(new float[] { -1, -1 }));
        assertThat(tree.getMass(tree.getLeftChild(node)), is(1));
        assertEquals(tree.getSequenceMap(tree.getPointIndex(tree.getLeftChild(node))).get(1L), 1);
        assertArrayEquals(new double[] { -1.0, 1.0 }, toDoubleArray(tree.getPointSum(node)), EPSILON);

        assertThat(tree.isLeaf(tree.getLeftChild(node)), is(true));
        assertThat(tree.pointStoreView.getNumericVector(tree.getPointIndex(tree.getLeftChild(node))),
                is(new float[] { -1, -1 }));
        assertThat(tree.getMass(tree.getLeftChild(node)), is(1));
        assertEquals(tree.getSequenceMap(tree.getPointIndex(tree.getLeftChild(node))).get(1L), 1);

        node = tree.getRightChild(node);
        expectedBox = new BoundingBox(new float[] { -1, 0 }).getMergedBox(new float[] { 1, 1 });
        assertThat(tree.getBox(node), is(expectedBox));
        assertThat(tree.getCutDimension(node), is(0));
        assertThat(tree.getCutValue(node), closeTo(0.5, EPSILON));
        assertThat(tree.getMass(node), is(3));

        assertArrayEquals(new double[] { 0.0, 2.0 }, toDoubleArray(tree.getPointSum(node)), EPSILON);

        assertThat(tree.isLeaf(tree.getRightChild(node)), is(true));
        assertThat(tree.pointStoreView.getNumericVector(tree.getPointIndex(tree.getRightChild(node))),
                is(new float[] { 1, 1 }));
        assertThat(tree.getMass(tree.getRightChild(node)), is(1));
        assertEquals(tree.getSequenceMap(tree.getPointIndex(tree.getRightChild(node))).get(2L), 1);

        NodeStoreSmall nodeStoreSmall = (NodeStoreSmall) tree.nodeStore;
        assert (nodeStoreSmall.getParentIndex(tree.getLeftChild(node)) == node);
        node = tree.getLeftChild(node);
        expectedBox = new BoundingBox(new float[] { -1, 0 }).getMergedBox(new float[] { 0, 1 });
        assertThat(tree.getBox(node), is(expectedBox));
        assertEquals(expectedBox.toString(), tree.getBox(node).toString());
        assertThat(tree.getCutDimension(node), is(0));
        assertThat(tree.getCutValue(node), closeTo(-0.5, EPSILON));
        assertThat(tree.getMass(), is(4));

        assertArrayEquals(new double[] { -1.0, 1.0 }, toDoubleArray(tree.getPointSum(node)), EPSILON);

        assertThat(tree.isLeaf(tree.getLeftChild(node)), is(true));
        assertThat(tree.pointStoreView.getNumericVector(tree.getPointIndex(tree.getLeftChild(node))),
                is(new float[] { -1, 0 }));
        assertThat(tree.getMass(tree.getLeftChild(node)), is(1));
        assertEquals(tree.getSequenceMap(tree.getPointIndex(tree.getLeftChild(node))).get(3L), 1);

        assertThat(tree.isLeaf(tree.getRightChild(node)), is(true));
        assertThat(tree.pointStoreView.getNumericVector(tree.getPointIndex(tree.getRightChild(node))),
                is(new float[] { 0, 1 }));
        assertThat(tree.getMass(tree.getRightChild(node)), is(1));
        assertEquals(tree.getSequenceMap(tree.getPointIndex(tree.getRightChild(node))).get(5L), 1);
    }

    @Test
    public void testDeletePointInvalid() {
        // specified sequence index does not exist
        assertThrows(IllegalArgumentException.class, () -> tree.deletePoint(2, 99));

        // point does not exist in tree
        assertThrows(IllegalArgumentException.class, () -> tree.deletePoint(7, 3));
    }

    @Test
    public void testUpdatesOnSmallBoundingBox() {
        // verifies on small bounding boxes random cuts and tree updates are functional
        PointStore pointStoreFloat = new PointStore.Builder().indexCapacity(10).capacity(10).currentStoreCapacity(10)
                .dimensions(1).build();
        RandomCutTree tree = RandomCutTree.builder().random(rng).pointStoreView(pointStoreFloat).build();

        List<Weighted<double[]>> points = new ArrayList<>();
        points.add(new Weighted<>(new double[] { 48.08 }, 0, 1L));
        points.add(new Weighted<>(new double[] { 48.08001 }, 0, 2L));

        pointStoreFloat.add(toFloatArray(points.get(0).getValue()), 0);
        pointStoreFloat.add(toFloatArray(points.get(1).getValue()), 1);
        tree.addPoint(0, points.get(0).getSequenceIndex());
        tree.addPoint(1, points.get(1).getSequenceIndex());
        assertNotEquals(pointStoreFloat.getNumericVector(0)[0], pointStoreFloat.getNumericVector(1)[0]);

        for (int i = 0; i < 10000; i++) {
            Weighted<double[]> point = points.get(i % points.size());
            tree.deletePoint(i % points.size(), point.getSequenceIndex());
            tree.addPoint(i % points.size(), point.getSequenceIndex());
        }
    }

    @Test
    public void testfloat() {
        float x = 110.13f;
        double sum = 0;
        int trials = 230000;
        for (int i = 0; i < trials; i++) {
            float z = (x * (trials - i + 1) - x);
            sum += z;
        }
        System.out.println(sum);
        for (int i = 0; i < trials - 1; i++) {
            float z = (x * (trials - i + 1) - x);
            sum -= z;
        }
        System.out.println(sum + " " + (double) x + " " + (sum <= (double) x));
        float[] possible = new float[trials];
        float[] alsoPossible = new float[trials];
        for (int i = 0; i < trials; i++) {
            possible[i] = x;
            alsoPossible[i] = (trials - i + 1) * x;
        }
        BoundingBox box = new BoundingBox(possible, alsoPossible);
        System.out.println("rangesum " + box.getRangeSum());
        double factor = 1.0 - 1e-16;
        System.out.println(factor);
        RandomCutTree tree = RandomCutTree.builder().dimension(trials).build();
        // tries both path
        tree.randomCut(factor, possible, box);
        tree.randomCut(1.0 - 1e-17, possible, box);

    }

    @ParameterizedTest
    @ValueSource(ints = { 100, 10000, 100000 })
    void testNodeStore(int size) {
        PointStore pointStoreFloat = new PointStore.Builder().indexCapacity(100).capacity(100).initialSize(100)
                .dimensions(2).build();
        tree = RandomCutTree.builder().random(rng).centerOfMassEnabled(true).pointStoreView(pointStoreFloat)
                .capacity(size).storeSequenceIndexesEnabled(true).storeParent(true).dimension(2).build();
        long seed = new Random().nextLong();
        System.out.println("seed :" + seed);
        Random rng = new Random(seed);
        for (int i = 0; i < 100; i++) {
            pointStoreFloat.add(new double[] { rng.nextDouble(), rng.nextDouble() }, 0L);
        }
        ArrayList<Weighted<Integer>> list = new ArrayList<>();
        for (int i = 0; i < 100; i++) {
            tree.addPoint(i, 0L);
            list.add(new Weighted<>(i, rng.nextFloat(), 0));
        }
        list.sort((o1, o2) -> Float.compare(o1.getWeight(), o2.getWeight()));
        for (int i = 0; i < 50; i++) {
            tree.deletePoint(list.remove(0).getValue(), 0L);
        }

        AbstractNodeStore nodeStore = tree.getNodeStore();
        for (int i = 0; i < 25; i++) {
            if (!tree.isLeaf(tree.getLeftChild(tree.getRoot()))) {
                assert (nodeStore.getParentIndex(tree.getLeftChild(tree.getRoot())) == tree.root);
            }
            ;
            if (!tree.isLeaf(tree.getRightChild(tree.getRoot()))) {
                assert (nodeStore.getParentIndex(tree.getRightChild(tree.getRoot())) == tree.root);
            }
            ;
            tree.deletePoint(list.remove(0).getValue(), 0L);
        }
    }

}
