// Copyright Verizon Media. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
package com.yahoo.container.plugin.mojo;

import org.apache.maven.artifact.Artifact;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.logging.Logger;

import static java.util.stream.Collectors.toList;

/**
 * Translates the scope of dependencies when constructing a test bundle.
 * Used by {@link Artifacts} to determine which artifacts that are provided by the runtime or must be included in the bundle.
 *
 * Dependencies of scope 'test' are by default translated to 'compile'. Dependencies of other scopes are kept as is.
 *
 * Default scope translation can be overridden through a comma-separated configuration string.
 * Each substring is a triplet on the form [groupId]:[artifactId]:[scope].
 * Scope translation overrides affects all transitive dependencies.
 * The ordering of the triplets determines the priority - only the first matching override will affect a given dependency.
 *
 * @author bjorncs
 */
class TestBundleDependencyScopeTranslator implements Artifacts.ScopeTranslator {

    private static final Logger log = Logger.getLogger(TestBundleDependencyScopeTranslator.class.getName());

    private final Map<Artifact, String> dependencyScopes;

    private TestBundleDependencyScopeTranslator(Map<Artifact, String> dependencyScopes) {
        this.dependencyScopes = dependencyScopes;
    }

    @Override public String scopeOf(Artifact artifact) { return Objects.requireNonNull(dependencyScopes.get(artifact)); }

    static TestBundleDependencyScopeTranslator from(Map<String, Artifact> dependencies, String rawConfig) {
        List<DependencyOverride> dependencyOverrides = toDependencyOverrides(rawConfig);
        Map<Artifact, String> dependencyScopes = new HashMap<>();
        for (Artifact dependency : dependencies.values()) {
            dependencyScopes.put(dependency, getScopeForDependency(dependency, dependencyOverrides, dependencies));
        }
        return new TestBundleDependencyScopeTranslator(dependencyScopes);
    }

    private static List<DependencyOverride> toDependencyOverrides(String rawConfig) {
        if (rawConfig == null || rawConfig.isBlank()) return List.of();
        return Arrays.stream(rawConfig.split(","))
                .map(String::strip)
                .filter(s -> !s.isBlank())
                .map(TestBundleDependencyScopeTranslator::toDependencyOverride)
                .collect(toList());
    }

    private static DependencyOverride toDependencyOverride(String overrideString) {
        String[] elements = overrideString.split(":");
        if (elements.length != 3) {
            throw new IllegalArgumentException("Invalid dependency override: " + overrideString);
        }
        return new DependencyOverride(elements[0], elements[1], elements[2]);
    }

    private static String stripVersionAndScope(String idInDependencyTrail) {
        int firstDelimiter = idInDependencyTrail.indexOf(':');
        int secondDelimiter = idInDependencyTrail.indexOf(':', firstDelimiter + 1);
        return idInDependencyTrail.substring(0, secondDelimiter);
    }

    private static String getScopeForDependency(
            Artifact dependency, List<DependencyOverride> overrides, Map<String, Artifact> otherArtifacts) {
        String oldScope = dependency.getScope();
        for (DependencyOverride override : overrides) {
            for (Artifact dependent : dependencyTrailOf(dependency, otherArtifacts)) {
                if (override.isForArtifact(dependent)) {
                    // This translation is not always correct for artifacts having 'runtime' scope dependencies.
                    // If such dependencies are overridden to 'compile' scope, its 'runtime' dependencies will get
                    // scope 'compile' instead of 'runtime'.
                    log.fine(() -> String.format(
                            "Overriding scope of '%s'; scope '%s' overridden to '%s'",
                            dependency.getId(), oldScope, override.scope));
                    return override.scope;
                }
            }
        }
        String newScope = defaultScopeTranslationOf(oldScope);
        log.fine(() -> String.format(
                "Using default scope translation for '%s'; scope '%s' translated to '%s'",
                dependency.getId(), oldScope, newScope));
        return newScope;
    }

    private static List<Artifact> dependencyTrailOf(Artifact artifact, Map<String, Artifact> otherArtifacts) {
        return artifact.getDependencyTrail().stream()
                .skip(1) // Maven project itself is the first entry
                .map(parentId -> otherArtifacts.get(stripVersionAndScope(parentId)))
                .filter(Objects::nonNull)
                .collect(toList());
    }

    private static String defaultScopeTranslationOf(String scope) {
        return scope.equals(Artifact.SCOPE_TEST) ? Artifact.SCOPE_COMPILE : scope;
    }

    private static class DependencyOverride {
        final String groupId;
        final String artifactId;
        final String scope;

        DependencyOverride(String groupId, String artifactId, String scope) {
            this.groupId = groupId;
            this.artifactId = artifactId;
            this.scope = scope;
        }

        boolean isForArtifact(Artifact artifact) {
            return artifact.getGroupId().equals(groupId) && artifact.getArtifactId().equals(artifactId);
        }
    }
}
