import assert from 'node:assert/strict';
import test from 'node:test';

import { providerTestStatusText } from '../modules/settings.js';
import {
    effortFieldForRoute,
    effortDescriptorForModel,
    effortSelectHtml,
} from '../modules/route_editor_primitives.js';

test('z.ai insufficient balance (429 code 1113) carries a plan hint, never a bare rate-limit cue', () => {
    const text = providerTestStatusText({ ok: false, error: 'No credits' });
    assert.match(text, /insufficient balance/i);
    assert.match(text, /top up the account or switch the plan/i);
    assert.doesNotMatch(text, /Rate limited/);
});

test('a GLM route descriptor renders exactly three identity tiers with named inheritance', () => {
    const glm = {
        carrier: 'reasoning_effort',
        tiers: ['low', 'high', 'max'],
        canonical_tiers: ['low', 'high', 'max'],
        absent_meaning: 'max',
    };
    const html = effortFieldForRoute('data-x', '', glm, { surfaceDefault: 'low · inherited from review effort' });
    const values = [...html.matchAll(/<option value="([^"]*)"/g)].map((m) => m[1]);
    assert.deepEqual(values, ['', 'low', 'high', 'max']);
    assert.match(html, /inherit · low · inherited from review effort/);
});

test('no selectable empty-value items beyond the single inherit choice', () => {
    const html = effortSelectHtml('data-y', 'high', 'review effort', {
        carrier: 'reasoning_effort', tiers: ['low', 'high', 'max'],
        canonical_tiers: ['low', 'high', 'max'], absent_meaning: 'provider_default',
    });
    const empties = [...html.matchAll(/<option value=""/g)].length;
    assert.equal(empties, 1);
    assert.match(html, /value="high" selected/);
});

test('a no-carrier route states it does not accept a tier instead of offering choices', () => {
    const html = effortFieldForRoute('data-z', 'high', {
        carrier: 'none', tiers: [], canonical_tiers: [], absent_meaning: 'off',
    });
    assert.match(html, /does not accept a reasoning-effort tier/);
    assert.match(html, /saved tier <code>high<\/code> is not carried/);
    assert.doesNotMatch(html, /<option/);
});

test('the absent-tier cost trap is named on z.ai-style routes', () => {
    const html = effortFieldForRoute('data-w', '', {
        carrier: 'none', tiers: [], canonical_tiers: [], absent_meaning: 'max',
    });
    assert.match(html, /absent tier bills at the provider maximum/);
});

test('descriptor lookup matches catalog values and ids, and yields null off-catalog', () => {
    const items = [
        { id: 'glm-5.3', value: 'openai-compatible::glm-5.3',
            effort_descriptor: { carrier: 'reasoning_effort', tiers: ['low', 'high', 'max'] } },
    ];
    assert.equal(effortDescriptorForModel(items, 'openai-compatible::glm-5.3').carrier, 'reasoning_effort');
    assert.equal(effortDescriptorForModel(items, 'glm-5.3').tiers.length, 3);
    assert.equal(effortDescriptorForModel(items, 'not/in-catalog'), null);
    assert.equal(effortDescriptorForModel(null, 'anything'), null);
});
