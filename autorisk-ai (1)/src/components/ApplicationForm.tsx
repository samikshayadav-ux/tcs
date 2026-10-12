import React from 'react';
import {
  FormValidationErrors,
  SampleProfileId,
  VehicleApplicationFormValues,
} from '../types';
import { SAMPLE_PROFILE_LIST } from '../data/mockData';

interface ApplicationFormProps {
  values: VehicleApplicationFormValues;
  errors: FormValidationErrors;
  selectedSampleId: SampleProfileId | null;
  onChange: (field: keyof VehicleApplicationFormValues, value: string) => void;
  onSubmit: (e: React.FormEvent) => void;
  onReset: () => void;
  onSelectSample: (sampleId: SampleProfileId) => void;
}

export const ApplicationForm: React.FC<ApplicationFormProps> = ({
  values,
  errors,
  selectedSampleId,
  onChange,
  onSubmit,
  onReset,
  onSelectSample,
}) => {
  const inputBorderClass = (field: keyof VehicleApplicationFormValues) =>
    errors[field]
      ? 'border-red-500 focus:border-red-600 bg-red-50/20'
      : 'border-slate-300 focus:border-blue-600 bg-white';

  const errorCount = Object.keys(errors).length;

  return (
    <section className="bg-white border border-slate-200 rounded-xl p-5 sm:p-6 h-full flex flex-col justify-between">
      <div>
        {/* Form Header */}
        <div className="pb-5 border-b border-slate-200">
          <div className="flex flex-wrap items-baseline justify-between gap-2">
            <h2 className="text-base font-semibold text-slate-900">
              Vehicle Insurance Application Form
            </h2>
            <span className="text-xs text-slate-500">
              Fields marked with <span className="text-red-600 font-semibold">*</span> are required
            </span>
          </div>
          <p className="text-xs text-slate-500 mt-1">
            Select a sample profile below to pre-fill the form or enter custom applicant and vehicle details, then click Analyze Risk.
          </p>

          {/* Three Sample Profile Buttons (Low, Medium, High) */}
          <div className="mt-4 pt-4 border-t border-slate-100">
            <p className="text-xs font-medium text-slate-700 mb-2">
              Load Illustrative Sample Profile:
            </p>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-2.5">
              {SAMPLE_PROFILE_LIST.map((profile) => {
                const isSelected = selectedSampleId === profile.id;
                return (
                  <button
                    key={profile.id}
                    type="button"
                    onClick={() => onSelectSample(profile.id)}
                    className={`px-3 py-2 rounded-lg border text-left transition-colors cursor-pointer ${
                      isSelected
                        ? 'bg-blue-50/80 border-blue-600 text-blue-950'
                        : 'bg-slate-50/70 border-slate-200 text-slate-700 hover:bg-slate-100 hover:border-slate-300'
                    }`}
                  >
                    <div className="flex items-center justify-between gap-1">
                      <span className="text-xs font-semibold whitespace-nowrap">
                        {profile.label}
                      </span>
                    </div>
                    <p className="text-xs text-slate-500 mt-0.5 truncate">
                      {profile.values.applicantName} · {profile.shortTag}
                    </p>
                  </button>
                );
              })}
            </div>
          </div>
        </div>

        {/* Validation Alert Banner */}
        {errorCount > 0 && (
          <div
            role="alert"
            className="mt-4 px-4 py-3 rounded-lg bg-red-50 border border-red-200 text-xs text-red-700 leading-relaxed"
          >
            <strong className="font-semibold">Validation Error:</strong> Please resolve{' '}
            {errorCount} highlighted {errorCount === 1 ? 'field' : 'fields'} below before analyzing risk.
          </div>
        )}

        <form onSubmit={onSubmit} noValidate className="mt-5 space-y-5">
          {/* Row 1: Applicant Name & Garaging State */}
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <div className="sm:col-span-2">
              <label
                htmlFor="applicantName"
                className="block text-xs font-medium text-slate-700 mb-1.5"
              >
                Primary Policyholder Full Name <span className="text-red-600">*</span>
              </label>
              <input
                id="applicantName"
                type="text"
                required
                placeholder="e.g. Elena Rostova"
                value={values.applicantName}
                onChange={(e) => onChange('applicantName', e.target.value)}
                className={`w-full px-3.5 py-2 text-sm text-slate-900 border rounded-lg focus:outline-none transition-colors ${inputBorderClass(
                  'applicantName'
                )}`}
              />
              {errors.applicantName && (
                <p className="mt-1 text-xs text-red-600">{errors.applicantName}</p>
              )}
            </div>

            <div>
              <label
                htmlFor="policyState"
                className="block text-xs font-medium text-slate-700 mb-1.5"
              >
                Garaging State
              </label>
              <select
                id="policyState"
                value={values.policyState}
                onChange={(e) => onChange('policyState', e.target.value)}
                className="w-full px-3.5 py-2 text-sm text-slate-900 bg-white border border-slate-300 rounded-lg focus:outline-none focus:border-blue-600"
              >
                <option value="CA">California (CA)</option>
                <option value="WA">Washington (WA)</option>
                <option value="NY">New York (NY)</option>
                <option value="TX">Texas (TX)</option>
                <option value="IL">Illinois (IL)</option>
              </select>
            </div>
          </div>

          {/* Row 2: Vehicle Age, Vehicle Type, Vehicle Usage */}
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <div>
              <label
                htmlFor="vehicleAge"
                className="block text-xs font-medium text-slate-700 mb-1.5"
              >
                Vehicle Age (Years) <span className="text-red-600">*</span>
              </label>
              <input
                id="vehicleAge"
                type="number"
                min={0}
                max={40}
                required
                placeholder="0–40"
                value={values.vehicleAge}
                onChange={(e) => onChange('vehicleAge', e.target.value)}
                className={`w-full px-3.5 py-2 text-sm text-slate-900 border rounded-lg font-mono-tabular focus:outline-none transition-colors ${inputBorderClass(
                  'vehicleAge'
                )}`}
              />
              {errors.vehicleAge && (
                <p className="mt-1 text-xs text-red-600">{errors.vehicleAge}</p>
              )}
            </div>

            <div>
              <label
                htmlFor="vehicleType"
                className="block text-xs font-medium text-slate-700 mb-1.5"
              >
                Vehicle Type <span className="text-red-600">*</span>
              </label>
              <select
                id="vehicleType"
                required
                value={values.vehicleType}
                onChange={(e) => onChange('vehicleType', e.target.value)}
                className={`w-full px-3.5 py-2 text-sm text-slate-900 border rounded-lg focus:outline-none transition-colors ${inputBorderClass(
                  'vehicleType'
                )}`}
              >
                <option value="">Select type...</option>
                <option value="Sedan">Sedan</option>
                <option value="SUV">SUV / Crossover</option>
                <option value="Truck">Light Truck</option>
                <option value="EV">Battery Electric (EV)</option>
                <option value="Performance">Performance / Coupe</option>
                <option value="Van">Commercial Van</option>
              </select>
              {errors.vehicleType && (
                <p className="mt-1 text-xs text-red-600">{errors.vehicleType}</p>
              )}
            </div>

            <div>
              <label
                htmlFor="vehicleUsage"
                className="block text-xs font-medium text-slate-700 mb-1.5"
              >
                Vehicle Usage <span className="text-red-600">*</span>
              </label>
              <select
                id="vehicleUsage"
                required
                value={values.vehicleUsage}
                onChange={(e) => onChange('vehicleUsage', e.target.value)}
                className={`w-full px-3.5 py-2 text-sm text-slate-900 border rounded-lg focus:outline-none transition-colors ${inputBorderClass(
                  'vehicleUsage'
                )}`}
              >
                <option value="">Select usage...</option>
                <option value="Personal Commute">Personal Commute</option>
                <option value="Pleasure / Weekend">Pleasure / Weekend</option>
                <option value="Business / Commercial">Business / Commercial</option>
                <option value="Rideshare / Delivery">Rideshare / Delivery</option>
              </select>
              {errors.vehicleUsage && (
                <p className="mt-1 text-xs text-red-600">{errors.vehicleUsage}</p>
              )}
            </div>
          </div>

          {/* Row 3: Annual Mileage, Driver Experience, Driver Age */}
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <div>
              <label
                htmlFor="annualMileage"
                className="block text-xs font-medium text-slate-700 mb-1.5"
              >
                Annual Mileage (mi/yr) <span className="text-red-600">*</span>
              </label>
              <input
                id="annualMileage"
                type="number"
                min={500}
                max={150000}
                required
                placeholder="e.g. 9500"
                value={values.annualMileage}
                onChange={(e) => onChange('annualMileage', e.target.value)}
                className={`w-full px-3.5 py-2 text-sm text-slate-900 border rounded-lg font-mono-tabular focus:outline-none transition-colors ${inputBorderClass(
                  'annualMileage'
                )}`}
              />
              {errors.annualMileage && (
                <p className="mt-1 text-xs text-red-600">{errors.annualMileage}</p>
              )}
            </div>

            <div>
              <label
                htmlFor="driverExperience"
                className="block text-xs font-medium text-slate-700 mb-1.5"
              >
                Driver Experience (Years) <span className="text-red-600">*</span>
              </label>
              <input
                id="driverExperience"
                type="number"
                min={0}
                max={80}
                required
                placeholder="0–80"
                value={values.driverExperience}
                onChange={(e) => onChange('driverExperience', e.target.value)}
                className={`w-full px-3.5 py-2 text-sm text-slate-900 border rounded-lg font-mono-tabular focus:outline-none transition-colors ${inputBorderClass(
                  'driverExperience'
                )}`}
              />
              {errors.driverExperience && (
                <p className="mt-1 text-xs text-red-600">{errors.driverExperience}</p>
              )}
            </div>

            <div>
              <label
                htmlFor="driverAge"
                className="block text-xs font-medium text-slate-700 mb-1.5"
              >
                Driver Age (Years) <span className="text-red-600">*</span>
              </label>
              <input
                id="driverAge"
                type="number"
                min={16}
                max={100}
                required
                placeholder="16–100"
                value={values.driverAge}
                onChange={(e) => onChange('driverAge', e.target.value)}
                className={`w-full px-3.5 py-2 text-sm text-slate-900 border rounded-lg font-mono-tabular focus:outline-none transition-colors ${inputBorderClass(
                  'driverAge'
                )}`}
              />
              {errors.driverAge && (
                <p className="mt-1 text-xs text-red-600">{errors.driverAge}</p>
              )}
            </div>
          </div>

          {/* Row 4: Previous Claims, At-Fault Claims, Vehicle Make/Model */}
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <div>
              <label
                htmlFor="previousClaims"
                className="block text-xs font-medium text-slate-700 mb-1.5"
              >
                Previous Claims (Total) <span className="text-red-600">*</span>
              </label>
              <input
                id="previousClaims"
                type="number"
                min={0}
                max={20}
                required
                placeholder="0–20"
                value={values.previousClaims}
                onChange={(e) => onChange('previousClaims', e.target.value)}
                className={`w-full px-3.5 py-2 text-sm text-slate-900 border rounded-lg font-mono-tabular focus:outline-none transition-colors ${inputBorderClass(
                  'previousClaims'
                )}`}
              />
              {errors.previousClaims && (
                <p className="mt-1 text-xs text-red-600">{errors.previousClaims}</p>
              )}
            </div>

            <div>
              <label
                htmlFor="atFaultClaims"
                className="block text-xs font-medium text-slate-700 mb-1.5"
              >
                At-Fault Claims <span className="text-red-600">*</span>
              </label>
              <input
                id="atFaultClaims"
                type="number"
                min={0}
                max={20}
                required
                placeholder="0–20"
                value={values.atFaultClaims}
                onChange={(e) => onChange('atFaultClaims', e.target.value)}
                className={`w-full px-3.5 py-2 text-sm text-slate-900 border rounded-lg font-mono-tabular focus:outline-none transition-colors ${inputBorderClass(
                  'atFaultClaims'
                )}`}
              />
              {errors.atFaultClaims && (
                <p className="mt-1 text-xs text-red-600">{errors.atFaultClaims}</p>
              )}
            </div>

            <div>
              <label
                htmlFor="vehicleMakeModel"
                className="block text-xs font-medium text-slate-700 mb-1.5"
              >
                Vehicle Make &amp; Model <span className="text-slate-400 font-normal">(Optional)</span>
              </label>
              <input
                id="vehicleMakeModel"
                type="text"
                placeholder="e.g. Volvo XC60 Recharge"
                value={values.vehicleMakeModel}
                onChange={(e) => onChange('vehicleMakeModel', e.target.value)}
                className="w-full px-3.5 py-2 text-sm text-slate-900 bg-white border border-slate-300 rounded-lg focus:outline-none focus:border-blue-600"
              />
            </div>
          </div>

          {/* Row 5: Coverage Package & Policy Deductible */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div>
              <label
                htmlFor="coverageTier"
                className="block text-xs font-medium text-slate-700 mb-1.5"
              >
                Coverage Package
              </label>
              <select
                id="coverageTier"
                value={values.coverageTier}
                onChange={(e) => onChange('coverageTier', e.target.value)}
                className="w-full px-3.5 py-2 text-sm text-slate-900 bg-white border border-slate-300 rounded-lg focus:outline-none focus:border-blue-600"
              >
                <option value="Liability Only">State Minimum Liability</option>
                <option value="Standard Split Limit">Standard 100/300/100</option>
                <option value="Comprehensive + Collision">Comprehensive + Collision</option>
                <option value="Premier Commercial">Premier Zero-Depreciation</option>
              </select>
            </div>

            <div>
              <label
                htmlFor="deductibleAmount"
                className="block text-xs font-medium text-slate-700 mb-1.5"
              >
                Policy Deductible (USD)
              </label>
              <select
                id="deductibleAmount"
                value={values.deductibleAmount}
                onChange={(e) => onChange('deductibleAmount', e.target.value)}
                className="w-full px-3.5 py-2 text-sm text-slate-900 bg-white border border-slate-300 rounded-lg font-mono-tabular focus:outline-none focus:border-blue-600"
              >
                <option value="500">$500</option>
                <option value="1000">$1,000</option>
                <option value="2500">$2,500</option>
              </select>
            </div>
          </div>

          {/* Form Actions: Reset & Analyze Risk */}
          <div className="pt-4 border-t border-slate-200 flex items-center justify-end gap-3">
            <button
              type="button"
              onClick={onReset}
              className="px-4 py-2 text-xs font-medium text-slate-700 bg-white border border-slate-300 rounded-lg hover:bg-slate-50 transition-colors whitespace-nowrap shrink-0 cursor-pointer"
            >
              Reset
            </button>
            <button
              type="submit"
              className="px-5 py-2 text-xs font-semibold text-white bg-blue-600 rounded-lg hover:bg-blue-700 transition-colors whitespace-nowrap shrink-0 cursor-pointer"
            >
              Analyze Risk
            </button>
          </div>
        </form>
      </div>
    </section>
  );
};
